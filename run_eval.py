import argparse

import os
import numpy as np
import random
import json
import time
import torch
import yaml
import hashlib

from data_utils import load_eval_data
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from dysco.custom_mixin import DyscoConfig
from dysco.custom_modeling_qwen3 import RescaleQwen3ForCausalLM
from dysco.custom_modeling_llama import RescaleLlamaForCausalLM

from tqdm import tqdm

_TOKENIZER_FOR_FILTERING = "models/Llama-3.1-8B-Instruct" # always use llama 3.1 8B tokenizer for filtering

def reset_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation_seed", type=int, default=23, help="seed for generation to minimize sampling variance")

    # data quantity
    parser.add_argument("--dataset", type=str, default="tom_tracking_0.5k")
    parser.add_argument("--test_size", type=int, default=-1, help="if -1, use all data")
    parser.add_argument("--no_chat_template", action="store_false", dest="use_chat_template", default=True, help="apply chat template to the input prompt")
    parser.add_argument("--stop_on_newline", action="store_true", dest="stop_on_newline", default=False, help="stop on newline")

    # query args
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=16384) # TODO: is max_model_len the context length?
    parser.add_argument("--think", action="store_true", default=False, help="enable thinking tokens for Qwen3")

    # control args
    parser.add_argument("--strip_thinking", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--disable_cache", dest="enable_cache", action="store_false", default=True)

    # add skip eval
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--auto_skip", action="store_true", default=False, help="skip run if both output and score files exist")

   # control decoding method
    parser.add_argument("--decoding_method", type=str, default="flash", choices=["flash", "dysco", "attnsharp"])
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for flash generation (only works with flash decoding)")
    parser.add_argument("--dysco_cfgs_path", type=str, default=None, help="path to dysco configs")
    parser.add_argument("--dysco_qrheads", type=str, default=None, help="qr heads")
    parser.add_argument("--dysco_top_k", type=int, default=None, help="top k")
    parser.add_argument("--dysco_top_p", type=float, default=None, help="top p")
    parser.add_argument("--dysco_strength", type=float, default=None, help="strength")
    parser.add_argument("--dysco_ctx_momentum", type=float, default=None, help="context momentum")
    parser.add_argument("--dysco_ctx_warmup", type=int, default=None, help="context warmup")
    parser.add_argument("--dysco_interv_warmup", type=str, default=None, help="intervention warmup")
    parser.add_argument("--dysco_rescale_template", action="store_true", default=False, help="rescale template")
    parser.add_argument("--dysco_static_rescaling", action="store_true", default=False, help="static rescaling")

    # not used for now
    parser.add_argument("--attention_logits_temperature", type=float, default=None, help="scale for attention sharpness") # attn sharp scales logits by a temperature 
    args = parser.parse_args()

    args.model = args.model.rstrip("/")
    saving_name = args.model.replace("/", "-")
    args.output_dir = os.path.join(args.output_dir, args.decoding_method, saving_name)

    return args



def get_output_path(args, dysco_config = None):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def hash_heads(heads):
        return hashlib.sha256(str(heads).encode()).hexdigest()[:8]

    if args.decoding_method in ["flash"]:
        output_filename = f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}t{args.temperature}p{args.top_p}_think{args.think}_sd{args.seed}gen{args.generation_seed}tsz{args.test_size}.json"
    elif args.decoding_method in ["dysco"]:
        output_filename = f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}t{args.temperature}p{args.top_p}_dsyscohead{hash_heads(dysco_config.dysco_qrheads)}tk{dysco_config.dysco_top_k}tp{dysco_config.dysco_top_p}s{dysco_config.dysco_strength}_m{dysco_config.dysco_ctx_momentum}ctxw{dysco_config.dysco_ctx_warmup}itw{dysco_config.dysco_interv_warmup}sctp{dysco_config.dysco_rescale_template}_think{args.think}_sd{args.seed}gen{args.generation_seed}tsz{args.test_size}.json"
    elif args.decoding_method in ["attnsharp"]:
        output_filename = f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}t{args.temperature}p{args.top_p}_attntemp{args.attention_logits_temperature}_think{args.think}_sd{args.seed}gen{args.generation_seed}tsz{args.test_size}.json"
    else:
        raise ValueError(f"Unknown decoding method: {args.decoding_method}")
    return os.path.join(args.output_dir, output_filename)


def filter_dataset_by_length(args, dataset):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_FOR_FILTERING)

    left_for_context = args.max_model_len - args.max_tokens - 16 # for generation prompt eos etc.
    new_dataset = []
    for ex in dataset:
        # Handle differently for MRCR dataset
        if args.dataset.startswith("mrcr"):
            conversation = ex["input_prompt"]
        else:
            conversation = [{"role": "user", "content": ex["input_prompt"]}]
        
        encoded = tokenizer.apply_chat_template(conversation=conversation, tokenize=True, add_generation_prompt=True)
        if len(encoded) <= left_for_context:
            new_dataset.append(ex)
    if len(new_dataset) != len(dataset):
        print(f"Filtered from {len(dataset)} to {len(new_dataset)}")
    return new_dataset

def obtain_template_sequence_mask(input_ids, template_sequences: list):
    # input_ids: (batch_size, max_length)
    # return a mask of all template sequences
    batch_size, max_length = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for template in template_sequences:
        template_len = len(template)
        # For each possible starting position
        for i in range(max_length - template_len + 1):
            # Check if template matches at position i
            # Compare input_ids[:, i:i+template_len] with template
            matches = torch.all(input_ids[:, i:i+template_len] == template.unsqueeze(0), dim=1)
            # Mark positions where template matches
            for b in range(batch_size):
                if matches[b]:
                    mask[b, i:i+template_len] = True

    return mask

def prepare_input_ids(ex, tokenizer, model_type, use_chat_template, enable_thinking):
    """Prepare input_ids from a single example.

    Returns:
        input_ids: tensor of shape [1, seq_len]
        input_prompt: the prompt used (either string or list of dicts)
    """
    # Check type of input prompt, used for MRCR dataset
    if isinstance(ex["input_prompt"], list) and use_chat_template == False:
        raise ValueError("Input prompt is already a list, chat template should be applied.")

    if isinstance(ex["input_prompt"], list):
        input_prompt = ex["input_prompt"]
        if "qwen3" in model_type:
            input_ids = tokenizer.apply_chat_template(input_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=enable_thinking)
        else:
            input_ids = tokenizer.apply_chat_template(input_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    else:
        if use_chat_template:
            input_prompt = [{"role": "user", "content": ex["input_prompt"]}]
            if "qwen3" in model_type:
                input_ids = tokenizer.apply_chat_template(input_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=enable_thinking)
            else:
                input_ids = tokenizer.apply_chat_template(input_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        else:
            input_prompt = ex["input_prompt"]
            input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

    return input_ids, input_prompt


def pad_batch_left(batch_input_ids, pad_token_id, device):
    """Left-pad a batch of input_ids and create attention masks.

    Args:
        batch_input_ids: list of 1D tensors of varying lengths
        pad_token_id: token id to use for padding
        device: device to place output tensors on

    Returns:
        padded_input_ids: tensor of shape [batch_size, max_len]
        attention_mask: tensor of shape [batch_size, max_len]
        input_lengths: list of original input lengths
    """
    max_len = max(ids.shape[0] for ids in batch_input_ids)
    padded_input_ids = []
    attention_mask = []
    input_lengths = []

    for input_ids in batch_input_ids:
        input_length = input_ids.shape[0]
        input_lengths.append(input_length)

        # Pad on the left
        padding_length = max_len - input_length
        padded_ids = torch.cat([
            torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype),
            input_ids
        ])
        mask = torch.cat([
            torch.zeros(padding_length, dtype=torch.long),
            torch.ones(input_length, dtype=torch.long)
        ])

        padded_input_ids.append(padded_ids)
        attention_mask.append(mask)

    # Stack into batch tensors
    padded_input_ids = torch.stack(padded_input_ids).to(device)
    attention_mask = torch.stack(attention_mask).to(device)

    return padded_input_ids, attention_mask, input_lengths


def setup_stop_token_ids(tokenizer, model, model_type, stop_on_newline):
    """Setup stop token IDs including newline tokens if requested.

    Returns:
        stop_token_ids: list of token ids or None if stop_on_newline is False
    """
    if not stop_on_newline:
        return None

    stop_token_ids = model.generation_config.eos_token_id
    stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
    stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
    stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + stop_token_ids))
    if "llama" in model_type:
        stop_token_ids.remove(tokenizer.unk_token_id)
    stop_token_ids = [x for x in stop_token_ids if x is not None]
    return stop_token_ids


def run_flash_generation(args, model, tokenizer, dataset):
    # Detect model type from config
    decoding_func = model.generate
    model_type = model.config.model_type.lower()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = setup_stop_token_ids(tokenizer, model, model_type, args.stop_on_newline)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline, "Batch size", args.batch_size)

    outputs = []

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Running generation", total=num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        # Prepare batch inputs
        batch_input_ids = []
        batch_prompts = []

        for ex in batch:
            input_ids, input_prompt = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
            batch_input_ids.append(input_ids[0])
            batch_prompts.append(ex["input_prompt"])

        # Pad the batch
        padded_input_ids, attention_mask, input_lengths = pad_batch_left(batch_input_ids, tokenizer.pad_token_id, model.device)
        max_len = padded_input_ids.shape[1]

        # Special sampling parameters for Qwen3 thinking mode
        if "qwen3" in model_type and args.think:
            decoding_kwargs = {
                "max_new_tokens": args.max_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "do_sample": True,
                "attention_mask": attention_mask,
                "pad_token_id": tokenizer.pad_token_id,
            }
        else:
            decoding_kwargs = {
                "max_new_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.temperature > 0,
                "attention_mask": attention_mask,
                "pad_token_id": tokenizer.pad_token_id,
            }

        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        generated_ids = decoding_func(padded_input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken

        # Decode each output in the batch
        for i, (gen_ids, input_len) in enumerate(zip(generated_ids, input_lengths)):
            # Extract only the generated tokens (skip the input)
            output_ids = gen_ids[max_len:]
            content = tokenizer.decode(output_ids, skip_special_tokens=True)

            global_idx = batch_start + i
            if global_idx < 2:
                print("-" * 50)
                print("<INPUT PROMPT>",)
                print(batch_prompts[i])
                print("<OUTPUT>",)
                print(content)

            outputs.append({
                "prompt": batch_prompts[i],
                "output": content,
                "success": True,
                "time_taken": time_taken / len(batch)  # Divide time among batch items
            })

    return outputs

def run_attnsharp_generation(args, model, tokenizer, dataset):
    ## set rescale config
    assert args.decoding_method in ["attnsharp"]

    decoding_func = model.dysco_generate
    model_type = model.config.model_type.lower()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = setup_stop_token_ids(tokenizer, model, model_type, args.stop_on_newline)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline, "Batch size", args.batch_size)

    # additional kwargs for attention sharpness
    additional_kwargs = {}
    additional_kwargs["use_attnsharp"] = True
    additional_kwargs["attention_logits_temperature"] = args.attention_logits_temperature

    outputs = []

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Running generation", total=num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        # Prepare batch inputs
        batch_input_ids = []
        batch_prompts = []

        for ex in batch:
            input_ids, input_prompt = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
            batch_input_ids.append(input_ids[0])
            batch_prompts.append(ex["input_prompt"])

        # Pad the batch
        padded_input_ids, attention_mask, input_lengths = pad_batch_left(batch_input_ids, tokenizer.pad_token_id, model.device)
        max_len = padded_input_ids.shape[1]

        # Special sampling parameters for Qwen3 thinking mode
        if "qwen3" in model_type and args.think:
            decoding_kwargs = {
                "max_new_tokens": args.max_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "do_sample": True,
                "attention_mask": attention_mask,
                "pad_token_id": tokenizer.pad_token_id,
            }
        else:
            decoding_kwargs = {
                "max_new_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.temperature > 0,
                "attention_mask": attention_mask,
                "pad_token_id": tokenizer.pad_token_id,
            }

        if stop_token_ids is not None:
            decoding_kwargs["eos_token_id"] = stop_token_ids
        decoding_kwargs.update(additional_kwargs)

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        generated_ids, generation_logging = decoding_func(padded_input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken

        # Decode each output in the batch
        for i, (gen_ids, input_len) in enumerate(zip(generated_ids, input_lengths)):
            # Extract only the generated tokens (skip the padded input)
            output_ids = gen_ids[max_len:]
            content = tokenizer.decode(output_ids, skip_special_tokens=True)

            global_idx = batch_start + i
            if global_idx < 2:
                print("-" * 50)
                print("<INPUT PROMPT>",)
                print(batch_prompts[i])
                print("<OUTPUT>",)
                print(content)

            outputs.append({
                "prompt": batch_prompts[i],
                "output": content,
                "success": True,
                "time_taken": time_taken / len(batch),
                "generation_logging": generation_logging
            })

    return outputs


def run_dysco_generation(args, model, tokenizer, dataset, dysco_config):
    ## set rescale config
    assert args.decoding_method in ["dysco"]
    assert args.batch_size == 1, "Dysco does not support batching"

    decoding_func = model.dysco_generate
    model_type = model.config.model_type.lower()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = setup_stop_token_ids(tokenizer, model, model_type, args.stop_on_newline)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline, "Batch size", args.batch_size)
    print("Dysco config", dysco_config.__dict__)

    outputs = []

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Running generation", total=num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        # Prepare batch inputs
        batch_input_ids = []
        batch_prompts = []

        for ex in batch:
            input_ids, input_prompt = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
            batch_input_ids.append(input_ids[0])
            batch_prompts.append(ex["input_prompt"])

        # Pad the batch
        padded_input_ids, attention_mask, input_lengths = pad_batch_left(batch_input_ids, tokenizer.pad_token_id, model.device)
        max_len = padded_input_ids.shape[1]

        # Special sampling parameters for Qwen3 thinking mode
        if "qwen3" in model_type and args.think:
            decoding_kwargs = {
                "max_new_tokens": args.max_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "do_sample": True,
                "dysco_config": dysco_config,
                "attention_mask": attention_mask,
                "pad_token_id": tokenizer.pad_token_id,
            }
        else:
            decoding_kwargs = {
                "max_new_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.temperature > 0,
                "dysco_config": dysco_config,
                "attention_mask": attention_mask,
                "pad_token_id": tokenizer.pad_token_id,
            }

        if stop_token_ids is not None:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        generated_ids, generation_logging = decoding_func(padded_input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken

        # Decode each output in the batch
        for i, (gen_ids, input_len) in enumerate(zip(generated_ids, input_lengths)):
            # Extract only the generated tokens (skip the padded input)
            output_ids = gen_ids[max_len:]
            content = tokenizer.decode(output_ids, skip_special_tokens=True)

            global_idx = batch_start + i
            if global_idx < 2:
                print("-" * 50)
                print("<INPUT PROMPT>",)
                print(batch_prompts[i])
                print("<OUTPUT>",)
                print(content)

            outputs.append({
                "prompt": batch_prompts[i],
                "output": content,
                "success": True,
                "time_taken": time_taken / len(batch),  # Divide time among batch items
                "generation_logging": generation_logging  # Same logging for all items in batch
            })

    return outputs



def build_dysco_config(args, tokenizer):
    # Load config from file
    if args.dysco_cfgs_path is not None:
        with open(args.dysco_cfgs_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    # Override with args if they are not None
    if args.dysco_qrheads is not None:
        config_dict['dysco_qrheads'] = args.dysco_qrheads
    if args.dysco_top_k is not None:
        config_dict['dysco_top_k'] = args.dysco_top_k if args.dysco_top_k > 0 else None
    if args.dysco_top_p is not None:
        config_dict['dysco_top_p'] = args.dysco_top_p if args.dysco_top_p > 0 else None
    if args.dysco_strength is not None:
        config_dict['dysco_strength'] = args.dysco_strength
    if args.dysco_ctx_momentum is not None:
        config_dict['dysco_ctx_momentum'] = args.dysco_ctx_momentum
    if args.dysco_ctx_warmup is not None:
        config_dict['dysco_ctx_warmup'] = args.dysco_ctx_warmup
    if args.dysco_interv_warmup is not None:  # override if not default value
        config_dict['dysco_interv_warmup'] = args.dysco_interv_warmup
    # For boolean values, override if they differ from the config value
    if config_dict.get('dysco_rescale_template', False) != args.dysco_rescale_template:
        config_dict['dysco_rescale_template'] = args.dysco_rescale_template
    if config_dict.get('dysco_static_rescaling', False) != args.dysco_static_rescaling:
        config_dict['dysco_static_rescaling'] = args.dysco_static_rescaling

    # process config dict
    if isinstance(config_dict['dysco_qrheads'], str):
        config_dict['dysco_qrheads'] = eval(config_dict['dysco_qrheads'])

    # intervention warmup steps
    if config_dict['dysco_interv_warmup'] == 'auto':
        if args.use_chat_template:
            # use all generation prompt tokens for warmup
            if "qwen3" in args.model.lower():
                dummy_input_ids = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=args.think)
                dummy_input_ids_no_gen_prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=True, add_generation_prompt=False, return_tensors="pt", enable_thinking=args.think)
            else:
                dummy_input_ids = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=True, add_generation_prompt=True, return_tensors="pt")
                dummy_input_ids_no_gen_prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=True, add_generation_prompt=False, return_tensors="pt")
            processed_warmup_steps = dummy_input_ids.shape[1] - dummy_input_ids_no_gen_prompt.shape[1]
        else:
            processed_warmup_steps = 4
    elif config_dict['dysco_interv_warmup'].isdigit():
        processed_warmup_steps = int(config_dict['dysco_interv_warmup']) # fixed number of steps
    else:
        raise ValueError(f"Unknown intervention warmup setting: {config_dict['dysco_interv_warmup']}")
    config_dict['dysco_interv_warmup'] = processed_warmup_steps
    # Create DyscoConfig instance
    dysco_config = DyscoConfig(
        dysco_qrheads=config_dict['dysco_qrheads'],
        dysco_top_k=config_dict['dysco_top_k'],
        dysco_top_p=config_dict['dysco_top_p'],
        dysco_strength=config_dict['dysco_strength'],
        dysco_ctx_momentum=config_dict['dysco_ctx_momentum'],
        dysco_ctx_warmup=config_dict['dysco_ctx_warmup'],
        dysco_interv_warmup=config_dict['dysco_interv_warmup'],
        dysco_rescale_template=config_dict['dysco_rescale_template'],
        dysco_static_rescaling=config_dict['dysco_static_rescaling'],
        dysco_template_seqs=config_dict['dysco_template_seqs'],
    )
    print("Dysco config", dysco_config.__dict__)
    return dysco_config


def main():
    args = _parse_args()

    config = AutoConfig.from_pretrained(args.model)
    model_type = config.model_type.lower()

    # Select appropriate model classes based on model type
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dysco_config = None if args.decoding_method != "dysco" else build_dysco_config(args, tokenizer)
    output_path = get_output_path(args, dysco_config)
    score_path = output_path.replace(".json", ".scores.json")
    print("Saving To", output_path)

    if args.decoding_method == "flash":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    elif args.decoding_method in ["dysco", "attnsharp"]:
        if "llama" in model_type:
            model = RescaleLlamaForCausalLM.from_pretrained(args.model, attn_implementation="flash_attention_2", device_map="auto", torch_dtype=torch.bfloat16)
        elif "qwen3" in model_type and "qwen3_moe" not in model_type:
            model = RescaleQwen3ForCausalLM.from_pretrained(args.model, attn_implementation="flash_attention_2", device_map="auto", torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Only 'llama' and 'qwen3' are supported for {args.decoding_method}.")

    dataset, eval_func = load_eval_data(args.dataset)

    # Auto skip if both output and score files exist
    if args.auto_skip and os.path.exists(output_path) and os.path.exists(score_path):
        print(f"Auto-skipping: both {output_path} and {score_path} exist")
        return

    # MRCR, graphwalks: don't filter
    if not args.dataset.startswith("mrcr") and not args.dataset.startswith("graphwalks") and not args.dataset.startswith("clipper"):
        dataset = filter_dataset_by_length(args, dataset)
    print(f"Dataset size: {len(dataset)}")
    # Don't shuffle and subsample for CLIPPER since we want to keep the pairs together
    if args.test_size > 0 and not args.dataset.startswith("clipper"):
        random.seed(args.seed)
        random.shuffle(dataset)
        dataset = dataset[:args.test_size]

    # batch querying
    if args.decoding_method == "flash":
        outputs = run_flash_generation(args, model, tokenizer, dataset)
    elif args.decoding_method == "dysco":
        outputs = run_dysco_generation(args, model, tokenizer, dataset, dysco_config)
    elif args.decoding_method == "attnsharp":
        outputs = run_attnsharp_generation(args, model, tokenizer, dataset)

    if args.skip_eval:
        print("Skipping eval") # everything is cached, so we can just return

    all_metrics = []
    saving_info = []
    
    for ex, output in zip(dataset, outputs):
        thinking_part = None
        if args.strip_thinking:
            if args.think and "</think>" in output["output"]:
                thinking_part = output["output"].rsplit("</think>", 1)[0].replace("<think>", "")
                output_part = output["output"].rsplit("</think>", 1)[1]
                output["output"] = output_part

        ex_saving = {
            "data": ex,
            "output": {"prompt": output["prompt"], "output": output["output"], "success": output["success"], "thinking_part": thinking_part,
            "generation_logging": output.get("generation_logging", None), "time_taken": output["time_taken"]},
        }
        if args.skip_eval:
            ex_saving["metric"] = None
        else:
            mets, _ = eval_func(output["output"], ex)
            all_metrics.append(mets)
            ex_saving["metric"] = mets
        saving_info.append(ex_saving)
    
    if args.skip_eval:
        output_content = {
            "args": args.__dict__,
            "saving_info": saving_info,
            "test_size": len(dataset),
        }
        with open(output_path, "w") as f:
            json.dump(output_content, f, indent=2)
        return
    
    # Special handling for CLIPPER: paired evaluation
    # CLIPPER data is organized as (TRUE[0], FALSE[0], TRUE[1], FALSE[1], ...)
    # A pair is correct only if both TRUE and FALSE are answered correctly
    if "clipper" in args.dataset:
        assert len(all_metrics) % 2 == 0, "CLIPPER must have even number of samples (paired)"

        # Compute paired accuracy
        num_pairs = len(all_metrics) // 2
        paired_correct = 0

        for i in range(num_pairs):
            true_idx = 2 * i  # TRUE samples at even indices
            false_idx = 2 * i + 1  # FALSE samples at odd indices

            true_correct = all_metrics[true_idx]["accuracy"]
            false_correct = all_metrics[false_idx]["accuracy"]

            # Pair is correct only if BOTH are correct
            if true_correct == 1 and false_correct == 1:
                paired_correct += 1

        # Compute metrics
        avg_metrics = {
            "accuracy": paired_correct / num_pairs,  # Main CLIPPER metric
        }

        # Add other metrics (averaged normally)
        for k in all_metrics[0].keys():
            if k != "accuracy":
                avg_metrics[k] = np.mean([x[k] for x in all_metrics])
    else:
        # Standard averaging for other datasets
        avg_metrics = {k: np.mean([x[k] for x in all_metrics]) for k in all_metrics[0].keys()}

    print([f"{k}: {v*100:.1f}" for k, v in avg_metrics.items()])

    # with open(output_filename, "w") as f:
        # for ex, prompt, output in zip(dataset, prompts, outputs):
    # avg_metrics = {k: np.mean([x[k] for x in all_metrics]) for k in all_metrics[0].keys()}
    output_content = {
        "args": args.__dict__,
        "saving_info": saving_info,
        "avg_metrics": avg_metrics,
        "test_size": len(dataset),
    }
    with open(output_path, "w") as f:
        json.dump(output_content, f, indent=2)
    with open(score_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)

if __name__=="__main__": 
    main()
