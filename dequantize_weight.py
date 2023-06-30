from falcontune.model import load_model
from falcontune.model.lora import load_adapter
import torch
from peft import LoraConfig, get_peft_model
import argparse


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
        replace_4bit_linear(child, names, bits, groupsize, quantlinear_class,
                            name + '.' + name1 if name != '' else name1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='falcon-7b-instruct-4bit')
    parser.add_argument('--falcon_ckpt', type=str, default='gptq_model-4bit-64g.safetensors')
    parser.add_argument('--backend', type=str, default='torch')

    parser.add_argument("--lora_r", default=8, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_alpha", default=16, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="Default: %(default)s")
    parser.add_argument("--target_modules", default="['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']", type=str, help="Target modules for LoRA.")

    args = parser.parse_args()

    falcon, tokenizer = load_model(model_name=args.model_name,
                        weights=args.falcon_ckpt,
                        backend=args.backend,
                        half=False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(falcon.config)

    model = get_peft_model(falcon, lora_config)

    print(model)
    for name, param in model.named_parameters():
        print(name, param.shape, param.min().item(), param.max().item(), param.mean().item())
