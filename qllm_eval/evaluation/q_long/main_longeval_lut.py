import argparse
import os
from tqdm import tqdm

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, dispatch_model, infer_auto_device_map

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from longeval.utils import maybe_monkey_patch, load_testcases, test_topics_one_sample, test_lines_one_sample

from playground.models.llama.modeling_llama import LlamaModel_use_block_sparse_attention_lut
from playground.attention.sparse_attention import set_static_attention_lut

# build model and tokenizer
def build_model_and_enc(model_path, use_flash_attn, kv_bit=16, kv_group_size=128):
    print(f"* Building model {model_path}")

    # weither trust remote code
    if 'chatglm' in model_path or 'mpt' in model_path or 'stable' in model_path:
        trust_remote_code = True
    else:
        trust_remote_code = False

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if use_flash_attn and 'chatglm' not in model_path and 'mpt' not in model_path:
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"
    elif use_flash_attn and 'mpt' in model_path:
        config.attn_config['attn_impl'] = 'triton'
    else:
        config._flash_attn_2_enabled = False
        config._attn_implementation = None

    config._attn_implementation = "eager"
    config._attn_implementation_internal = "eager"

    # add the kv quantization parameters
    config.kv_bit = kv_bit
    config.kv_group_size = kv_group_size

    # load tokenizer
    if 'mpt' in model_path or 'stable' in model_path:
        use_fast = True
    else:
        use_fast = False
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=trust_remote_code)

    # load model
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    # kwargs = {"torch_dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=trust_remote_code, **kwargs)
    # model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code, **kwargs)
    return model, enc


def get_output_dir(args):
    path = args.model_name_or_path

    if path[-1] == "/":
        path = path[:-1]
    name = path.split("/")[-1]

    output_dir = f"{args.test_dir}/{args.task}/predictions/{name}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"output to {output_dir}")
    return output_dir

def longeval_test(model, tokenizer, output_dir, args):
    if args.task == "topics":
        for num_topics in [5, 10, 15, 20, 25]:
            print(f"************ Start testing {num_topics} topics per prompt ***********")
            avg_length = 0

            test_file = os.path.join(args.test_dir, f"topics/testcases/{num_topics}_topics.jsonl")
            output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
            
            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                _, prompt_length, summary = test_topics_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)

            print(f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length} ************")
            if args.eval_shortest_only:
                break
            
    elif args.task == "lines":
        accuracy_list = []
        for num_lines in args.num_lines:
        # for num_lines in [600,680]:
            print(f"************ Start testing {num_lines} lines per LRT prompt ************")
            test_file = os.path.join(args.test_dir, f"lines/testcases/{num_lines}_lines.jsonl")
            
            output_file = os.path.join(output_dir, f"{num_lines}_response.txt")
            num_correct = 0
            avg_length = 0

            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                correct, prompt_length, summary = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)
                num_correct += correct
            accuracy = num_correct / len(test_cases)

            with open(output_file, "a+") as f:
                f.write(f"Accuracy: {accuracy}")

            print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
            accuracy_list.append(accuracy)
            if args.eval_shortest_only:
                break
        return accuracy_list
    else:
        print(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default='/share/datasets/public_models/lmsys_vicuna-7b-v1.5-16k', help="model path")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--task", type=str, default='lines', help="Which evaluation task to use. currently support [topics, lines]")
    parser.add_argument("--max_gpu_memory", type=int, default=80, help="max per gpu memory in GiB. A100 is 40 or 80.")
    parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
    parser.add_argument("--test_dir", type=str, default="evaluation", help="Directory of the testcases")
    parser.add_argument("--framework", type=str, default=None, help="Framework for serving")
    parser.add_argument("--num_lines", type=int, nargs="+", default=[170], help="Number of lines per prompt")
    # quantization config
    parser.add_argument("--kv_group_size", type=int, default=128)
    parser.add_argument("--kv_bit", type=int, default=16)
    # sparse attention config
    parser.add_argument("--lut_path", type=str, default=None, help="Path to the LUT file")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for the LUT")


    args = parser.parse_args()

    maybe_monkey_patch(args)
    output_dir = get_output_dir(args)

    model, tokenizer = build_model_and_enc(args.model_name_or_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

    # sparse model
    if args.lut_path is not None:
        print("Using lut from {}, with block size {}".format(args.lut_path, args.block_size))
        model.model.use_block_sparse_attention_lut = LlamaModel_use_block_sparse_attention_lut.__get__(model.model)
        model.model.use_block_sparse_attention_lut()
        set_static_attention_lut(args.lut_path, None, model.model.layers, args.block_size)

    # with init_empty_weights():
    # model = load_checkpoint_and_dispatch(model, checkpoint=args.model_name_or_path, device_map='auto', no_split_module_classes=["LlamaDecoderLayer"])

    accuracy_list = longeval_test(model, tokenizer, output_dir, args)
