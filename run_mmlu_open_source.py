import argparse
import json
import os
import time
# import utils

import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from falcontune.data import make_prompt
from falcontune.model import load_model
from falcontune.model.lora import load_adapter
from falcontune.model.utils import model_to_half
from peft import LoraConfig, get_peft_model

TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions']

choices = ["A", "B", "C", "D"]


class AMPWrapper:
    def __init__(self, model, options=None):
        self.model = model
        self.options = options
        if self.options is None:
            self.options = {'enabled': True, 'device_type': 'cuda'}

    def autocast_forward(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_forward(*args, **kwargs)

    def autocast_generate(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_generate(*args, **kwargs)

    def apply_forward(self):
        self.model.non_autocast_forward = self.model.forward
        self.model.forward = self.autocast_forward

    def apply_generate(self):
        self.model.non_autocast_generate = self.model.generate
        self.model.generate = self.autocast_generate


def format_output(raw_output):
    return raw_output.split("### Response:")[1].strip()


def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc / total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n 
#     return input_ids[-len(stop_ids)]

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k: input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens


def load(ckpt_dir, model_type):
    n_gpus = torch.cuda.device_count()

    if model_type == 'llama':
        # we use tensor parallel for loading llama
        tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")

        model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage=True, torch_dtype=torch.float32)

        # Quantize
        print(model)
        allow_name = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
        for name, param in model.named_parameters():
            if any(bn in name for bn in block_name):
                continue
            if any(an in name for an in allow_name):
                print("=================================")
                print(name, param.mean().item())
                quantized_weight = utils.quantize_weight(param, clip_val=None, num_bits=args.num_bits)
                param.data = quantized_weight
                print(name, param.mean().item())

        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    else:
        # mpt-30b's tokenizer only has the fast version
        use_fast = "mosaicml/mpt-30b" in ckpt_dir
        # however, tensor parallel for running falcon will occur bugs
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=use_fast, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir,
                                                     device_map='auto',
                                                     torch_dtype=torch.float32,
                                                     trust_remote_code=True)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    # print(answers)
    answers = [answer[-1] for answer in answers]
    return answers


def main(ckpt_dir: str, param_size: str, model_type: str):
    run_results = {}
    output_filename = 'run_results_%s_%sb.json' % (model_type, param_size)

    # model, tokenizer = load(ckpt_dir, model_type)
    model, tokenizer = load_model(
        args.model,
        args.weights,
        backend=args.backend)

    if args.lora_apply_dir is not None:
        model = load_adapter(model, lora_apply_dir=args.lora_apply_dir)

    tokenizer.padding_side = 'left'
    model = model.to('cuda')
    if getattr(model, 'loaded_in_4bit', False):
        model_to_half(model)

    print(model)
    wrapper = AMPWrapper(model)
    wrapper.apply_generate()

    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({'prompt': prompt, 'answer': label})

        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers': pred_answers, 'gold_answers': gold_answers}
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='tiiuae/falcon-7b')
    parser.add_argument('--param_size', type=str, default='7')
    parser.add_argument('--model_type', type=str, default='falcon')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--num_bits', type=int, default=4)
    parser.add_argument('--reduced_rank', type=int, default=8)
    parser.add_argument('--act_quant', action='store_true')
    parser.add_argument('--model', type=str, default='falcon-7b-instruct-4bit')
    parser.add_argument('--weights', type=str, default='gptq_model-4bit--1g.safetensors')
    parser.add_argument('--backend', type=str, default='torch', required=False, help='Change the default backend.')
    parser.add_argument("--lora_r", default=8, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_alpha", default=16, type=int, help="Default: %(default)s")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="Default: %(default)s")
    parser.add_argument("--target_modules", default="['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']",
                        type=str, help="Target modules for LoRA.")
    parser.add_argument("--lora_apply_dir", default="./lora_adapter",
                        type=str, help="path to lora adapter")
    args = parser.parse_args()

    main(args.ckpt_dir, args.param_size, args.model_type)

    """
    python run_mmlu_open_source.py --ckpt_dir tiiuae/falcon-7b --model_type falcon --num_bits 4 --reduced_rank 8 --act_quant
    """
