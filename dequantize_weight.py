from falcontune.model import load_model
from falcontune.model.lora import load_adapter
import torch
from peft import LoraConfig, get_peft_model
import argparse
from falcontune.model import lora
from transformers import AutoModelForCausalLM


def find_4bit_layers(module, layers=[lora.Linear4bitLt], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_4bit_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def svd_init(module, names, bits, groupsize, quantlinear_class, name=''):
    if isinstance(module, quantlinear_class):
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, quantlinear_class(bits, groupsize, tmp.in_features, tmp.out_features))

    for name1, child in module.named_children():
        svd_init(child, names, bits, groupsize, quantlinear_class,
                            name + '.' + name1 if name != '' else name1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='falcon-7b-instruct-4bit')
    parser.add_argument('--falcon_ckpt_q', type=str, default='gptq_model-4bit-64g.safetensors')
    parser.add_argument('--falcon_ckpt_f', type=str, default='tiiuae/falcon-7b')
    parser.add_argument('--backend', type=str, default='torch')

    parser.add_argument("--lora_r", default=8, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_alpha", default=16, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="Default: %(default)s")
    parser.add_argument("--target_modules", default="['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']",
                        type=str, help="Target modules for LoRA.")

    args = parser.parse_args()

    falcon, tokenizer = load_model(model_name=args.model_name,
                                   weights=args.falcon_ckpt_q,
                                   backend=args.backend,
                                   half=False)


    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dmodel = get_peft_model(falcon, lora_config)

    fmodel = AutoModelForCausalLM.from_pretrained(args.falcon_ckpt_f,
                                                  device_map='auto',
                                                  torch_dtype=torch.float,
                                                  trust_remote_code=True)

    print(dmodel)
    print(fmodel)

    print(dmodel.state_dict().keys())
    print("======================================")
    print(fmodel.state_dict().keys(), len(fmodel.state_dict().keys()))


    # svd_init(
    #     model,
    #     layers,
    # )
