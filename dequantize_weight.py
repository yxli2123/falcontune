from falcontune.model import load_model
from falcontune.model.lora import load_adapter
import torch
from peft import LoraConfig, get_peft_model
import argparse
from falcontune.model import lora
from transformers import AutoModelForCausalLM
import os


def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param    reduced_rank: the final rank
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    print(f"W: ({H},{W}) | Rank: {rank} | U:{U.shape} | S:{S.shape} | Vh:{Vh.shape}")
    print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
    print(f"L: {L.shape} | R: {R.shape}")

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'reduced_rank': reduced_rank}


def find_4bit_layers(module, layers=None, name=''):
    if layers is None:
        layers = [lora.Linear4bitLt]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_4bit_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def svd_init(module, name=''):

    for attr in dir(module):
        tmp = getattr(module, attr)
        name_sub = name + '.' + attr if name != '' else attr
        if isinstance(tmp, lora.Linear4bitLt):
            dequantized_weight = tmp.dequantize_base()
            name_in_full_model = (name_sub + ".weight").replace("base_model.model.", "")
            original_weight = fmodel_dict[name_in_full_model].T
            error = (dequantized_weight - original_weight).pow(2).mean().sqrt().item()
            print(name_in_full_model, dequantized_weight.shape, original_weight.shape, error)

            result = low_rank_decomposition(original_weight - dequantized_weight, reduced_rank=args.lora_r)
            L, R = result['L'], result['R']

            tmp.lora_A.default.weight.data = L
            tmp.lora_B.default.weight.data = R


    for name1, child in module.named_children():
        svd_init(child, name + '.' + name1 if name != '' else name1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='falcon-7b-instruct-4bit')
    parser.add_argument('--falcon_ckpt_q', type=str, default='gptq_model-4bit-64g.safetensors')
    parser.add_argument('--falcon_ckpt_f', type=str, default='tiiuae/falcon-7b')
    parser.add_argument('--backend', type=str, default='torch')

    parser.add_argument("--lora_r", default=8, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_alpha", default=16, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="Default: %(default)s")
    parser.add_argument("--target_modules", default="query_key_value, dense, dense_h_to_4h, dense_4h_to_h",
                        type=str, help="Target modules for LoRA.")

    parser.add_argument('--lora_out_dir', type=str, default='./output/lora')

    args = parser.parse_args()

    falcon, tokenizer = load_model(model_name=args.model_name,
                                   weights=args.falcon_ckpt_q,
                                   backend=args.backend,
                                   half=False)
    print("========>Loading quantized model")
    os.system("nvidia-smi")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dmodel = get_peft_model(falcon, lora_config)

    print("========>Adding LoRA")
    os.system("nvidia-smi")

    fmodel = AutoModelForCausalLM.from_pretrained(args.falcon_ckpt_f,
                                                  device_map='auto',
                                                  torch_dtype=torch.float,
                                                  trust_remote_code=True)
    fmodel_dict = fmodel.state_dict()
    print(fmodel_dict.keys())
    del fmodel

    print("========>Loading Full-precision Model")
    os.system("nvidia-smi")

    svd_init(dmodel)

    print("========>Replace LoRA with SVD")
    os.system("nvidia-smi")

    for name, param in dmodel.named_parameters():
        print(name, param.mean())

    dmodel.print_trainable_parameters()
    dmodel.save_pretrained(args.lora_out_dir)

    print("Model saved")

