import math
import numpy as np

import torch
import torch.nn as nn


def replace_4bit_linear(module, names, bits, groupsize, quantlinear_class, name=''):
    if isinstance(module, quantlinear_class):
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, quantlinear_class(bits, groupsize, tmp.in_features, tmp.out_features))

    for name1, child in module.named_children():
        replace_4bit_linear(child, names, bits, groupsize, quantlinear_class, name + '.' + name1 if name != '' else name1)


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


class QuantLinearBase(nn.Module):
    framework = ''

    def __init__(self, bits, groupsize, infeatures, outfeatures):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.maxq = 2 ** self.bits - 1

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor([[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0], ],
                                   dtype=torch.int32).reshape(1, 3, 12)

    def dequantize_base(self):
        self.qzeros = self.qzeros.to(self.qweight.device)
        self.scales = self.scales.to(self.qweight.device)
        self.g_idx = self.g_idx.to(self.qweight.device)
        self.wf = self.wf.to(self.qweight.device)

        if self.bits in [2, 4, 8]:
            zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                                              self.wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                                               self.wf.unsqueeze(-1)).to(
                torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(weight, (2 ** self.bits) - 1, out=weight)
        elif self.bits == 3:
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1,
                                                                                                      12)
            zeros = (zeros >> self.wf.unsqueeze(0))
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = torch.cat([zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2)

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1,
                                                                                                          12, -1)
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        else:
            raise NotImplemented('bits in [2, 3, 4, 8]')

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        # num_itr = self.g_idx.shape[0] // x.shape[-1]
        num_itr = self.g_idx.shape[0] // self.infeatures

        if num_itr == 1:
            weights = (self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()]))
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim:(i + 1) * num_dim]
                weight_i = weight[:, i * num_dim:(i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim:(i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim:(i + 1) * num_dim]
                weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
            weights = torch.cat(weights, dim=1)

        return weights

    # def pack(self, linear, scales, zeros, g_idx=None):
    #     self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
    #
    #     scales = scales.t().contiguous()
    #     zeros = zeros.t().contiguous()
    #     scale_zeros = zeros * scales
    #     self.scales = scales.clone().half()
    #     if linear.bias is not None:
    #         self.bias = linear.bias.clone().half()
    #
    #     intweight = []
    #     for idx in range(self.infeatures):
    #         intweight.append(torch.round(
    #             (linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(
    #             torch.int)[:, None])
    #     intweight = torch.cat(intweight, dim=1)
    #     intweight = intweight.t().contiguous()
    #     intweight = intweight.numpy().astype(np.uint32)
    #     qweight = np.zeros(
    #         (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
    #     )
    #     i = 0
    #     row = 0
    #     while row < qweight.shape[0]:
    #         if self.bits in [2, 4, 8]:
    #             for j in range(i, i + (32 // self.bits)):
    #                 qweight[row] |= intweight[j] << (self.bits * (j - i))
    #             i += 32 // self.bits
    #             row += 1
    #         elif self.bits == 3:
    #             for j in range(i, i + 10):
    #                 qweight[row] |= intweight[j] << (3 * (j - i))
    #             i += 10
    #             qweight[row] |= intweight[i] << 30
    #             row += 1
    #             qweight[row] |= (intweight[i] >> 2) & 1
    #             i += 1
    #             for j in range(i, i + 10):
    #                 qweight[row] |= intweight[j] << (3 * (j - i) + 1)
    #             i += 10
    #             qweight[row] |= intweight[i] << 31
    #             row += 1
    #             qweight[row] |= (intweight[i] >> 1) & 0x3
    #             i += 1
    #             for j in range(i, i + 10):
    #                 qweight[row] |= intweight[j] << (3 * (j - i) + 2)
    #             i += 10
    #             row += 1
    #         else:
    #             raise NotImplementedError("Only 2,3,4,8 bits are supported.")
    #
    #     qweight = qweight.astype(np.int32)
    #     self.qweight = torch.from_numpy(qweight)
    #
    #     zeros -= 1;
    #     zeros = zeros.numpy().astype(np.uint32)
    #     qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
    #     i = 0
    #     col = 0
    #     while col < qzeros.shape[1]:
    #         if self.bits in [2, 4, 8]:
    #             for j in range(i, i + (32 // self.bits)):
    #                 qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
    #             i += 32 // self.bits
    #             col += 1
    #         elif self.bits == 3:
    #             for j in range(i, i + 10):
    #                 qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
    #             i += 10
    #             qzeros[:, col] |= zeros[:, i] << 30
    #             col += 1
    #             qzeros[:, col] |= (zeros[:, i] >> 2) & 1
    #             i += 1
    #             for j in range(i, i + 10):
    #                 qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
    #             i += 10
    #             qzeros[:, col] |= zeros[:, i] << 31
    #             col += 1
    #             qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
    #             i += 1
    #             for j in range(i, i + 10):
    #                 qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
    #             i += 10
    #             col += 1
    #         else:
    #             raise NotImplementedError("Only 2,3,4,8 bits are supported.")
    #
    #     qzeros = qzeros.astype(np.int32)
    #     self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        raise NotImplementedError
