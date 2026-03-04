import argparse
import hashlib
import os
import numpy as np
import random
import json
import time
import yaml

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

from data_utils import load_eval_data
from dysco.custom_mixin import RescaleConfig
from dysco.custom_modeling_llama import RescaleLlamaForCausalLM
from dysco.custom_modeling_qwen3 import RescaleQwen3ForCausalLM
from dysco.custom_modeling_qwen3_moe import RescaleQwen3MoeForCausalLM

_TOKENIZER_FOR_FILTERING = "models/Llama-3.1-8B-Instruct"

# Maps HF model_type -> (BaseClass, RescaleClass)
_MODEL_CLASSES = {
    "llama":     (LlamaForCausalLM,     RescaleLlamaForCausalLM),
    "qwen3_moe": (Qwen3MoeForCausalLM,  RescaleQwen3MoeForCausalLM),
    "qwen3":     (Qwen3ForCausalLM,     RescaleQwen3ForCausalLM),
}


def reset_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation_seed", type=int, default=23)

    # data
    parser.add_argument("--dataset", type=str, default="tom_tracking_0.5k")
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--no_chat_template", action="store_false", dest="use_chat_template", default=True)
    parser.add_argument("--stop_on_newline", action="store_true", default=False)

    # model
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--think", action="store_true", default=False)

    # control
    parser.add_argument("--strip_thinking", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--disable_cache", dest="enable_cache", action="store_false", default=True)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--auto_skip", action="store_true", default=False)

    # decoding method
    parser.add_argument("--decoding_method", type=str, default="flash",
                        choices=["flash", "dysco", "attnsharp"])
    # DySCO config (YAML + overrides)
    parser.add_argument("--dysco_cfgs_path", type=str, default=None, help="path to dysco configs")
    parser.add_argument("--dysco_qrheads", type=str, default=None, help="qr heads")
    parser.add_argument("--dysco_top_k", type=int, default=None, help="top k")
    parser.add_argument("--dysco_top_p", type=float, default=None, help="top p")
    parser.add_argument("--dysco_strength", type=float, default=None, help="strength")
    parser.add_argument("--dysco_decay_factor", type=float, default=None, help="decay factor")
    parser.add_argument("--dysco_ctx_warmup", type=int, default=None, help="context warmup")
    parser.add_argument("--dysco_interv_warmup", type=str, default=None, help="intervention warmup")
    parser.add_argument("--dysco_rescale_template", action="store_true", default=False, help="rescale template")
    parser.add_argument("--dysco_static_rescaling", action="store_true", default=False, help="static rescaling")

    # attnsharp
    parser.add_argument("--attention_logits_temperature", type=float, default=None)

    args = parser.parse_args()
    args.model = args.model.rstrip("/")
    saving_name = args.model.replace("/", "-")
    args.output_dir = os.path.join(args.output_dir, args.decoding_method, saving_name)
    return args


def hash8(s):
    return hashlib.md5(str(s).encode()).hexdigest()[:8]


def get_output_path(args, rescale_config=None):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.decoding_method == "flash":
        output_filename = (
            f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}"
            f"t{args.temperature}p{args.top_p}"
            f"_think{args.think}_{args.seed}and{args.generation_seed}"
            f"_testsz{args.test_size}.json"
        )
    elif args.decoding_method == "attnsharp":
        output_filename = (
            f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}"
            f"t{args.temperature}p{args.top_p}"
            f"_attnsharp{args.attention_logits_temperature}"
            f"_think{args.think}_{args.seed}and{args.generation_seed}"
            f"_testsz{args.test_size}.json"
        )
    elif args.decoding_method == "dysco":
        cfg = rescale_config
        head_hash = hash8(cfg.selected_heads)
        dynamic_tag = "dynamic" if cfg.dynamic_rescale else "static"
        output_filename = (
            f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}"
            f"t{args.temperature}p{args.top_p}"
            f"_{dynamic_tag}rescalehead{head_hash}"
            f"k{cfg.top_k}p{cfg.top_p}s{cfg.strength}df{cfg.decay_factor}"
            f"_ctxwarm{cfg.context_warmup_steps}intwarm{cfg.intervention_warmup_steps}"
            f"scaletemp{args.dysco_rescale_template}"
            f"_think{args.think}_{args.seed}and{args.generation_seed}"
            f"_testsz{args.test_size}.json"
        )
    else:
        raise ValueError(f"Unknown decoding method: {args.decoding_method}")

    return os.path.join(args.output_dir, output_filename)


def detect_model_type(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type.lower()
    # Match against known types (order matters: qwen3_moe before qwen3)
    for key in ["llama", "qwen3_moe", "qwen3"]:
        if key in model_type:
            return key
    raise ValueError(f"Unsupported model type: {model_type}")


def build_rescale_config(args, tokenizer, model_type):
    """Load YAML config + CLI overrides -> RescaleConfig."""
    # Load base config from YAML
    if args.dysco_cfgs_path:
        with open(args.dysco_cfgs_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # CLI overrides (non-None values take precedence)
    if args.dysco_qrheads is not None:
        cfg["selected_heads"] = args.dysco_qrheads
    if args.dysco_top_k is not None:
        if args.dysco_top_k > 0:
            cfg["top_k"] = args.dysco_top_k
        else:
            cfg.pop("top_k", None)  # disable top_k (use top_p only)
    if args.dysco_top_p is not None:
        if args.dysco_top_p > 0:
            cfg["top_p"] = args.dysco_top_p
        else:
            cfg.pop("top_p", None)  # disable top_p (use top_k only)
    if args.dysco_strength is not None:
        cfg["strength"] = args.dysco_strength
    if args.dysco_decay_factor is not None:
        cfg["decay_factor"] = args.dysco_decay_factor
    if args.dysco_ctx_warmup is not None:
        cfg["context_warmup_steps"] = args.dysco_ctx_warmup
    if args.dysco_interv_warmup is not None:
        cfg["intervention_warmup"] = args.dysco_interv_warmup
    if args.dysco_rescale_template:
        cfg["scale_template_tokens"] = True
    if args.dysco_static_rescaling:
        cfg["dynamic_rescale"] = False

    # Parse selected_heads string -> list of tuples
    selected_heads = eval(cfg["selected_heads"])

    # Compute intervention_warmup_steps
    intervention_warmup = cfg.get("intervention_warmup", "auto")
    if intervention_warmup == "auto":
        if args.use_chat_template:
            if "qwen3" in model_type:
                dummy = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt",
                    enable_thinking=args.think)
                dummy_no_gen = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=False, return_tensors="pt",
                    enable_thinking=args.think)
            else:
                dummy = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt")
                dummy_no_gen = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=False, return_tensors="pt")
            intervention_warmup_steps = dummy.shape[1] - dummy_no_gen.shape[1]
        else:
            intervention_warmup_steps = 2
    elif str(intervention_warmup).isdigit():
        intervention_warmup_steps = int(intervention_warmup)
    else:
        raise ValueError(f"Unknown intervention warmup setting: {intervention_warmup}")

    # Template sequences (None when rescale_template is enabled — no masking)
    template_sequences = None
    if not cfg.get("scale_template_tokens", False):
        raw_templates = cfg.get("template_sequences", [])
        if raw_templates:
            template_sequences = [torch.LongTensor(seq) for seq in raw_templates]

    rescale_config = RescaleConfig(
        selected_heads=selected_heads,
        top_k=cfg.get("top_k"),
        top_p=cfg.get("top_p"),
        strength=cfg["strength"],
        decay_factor=cfg["decay_factor"],
        context_warmup_steps=cfg.get("context_warmup_steps", 0),
        intervention_warmup_steps=intervention_warmup_steps,
        dynamic_rescale=cfg.get("dynamic_rescale", True),
        template_sequences=template_sequences,
    )

    return rescale_config



def filter_dataset_by_length(args, dataset):
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_FOR_FILTERING)
    left_for_context = args.max_model_len - args.max_tokens - 16
    new_dataset = []
    for ex in dataset:
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


def prepare_input_ids(ex, tokenizer, model_type, use_chat_template, think):
    if isinstance(ex["input_prompt"], list):
        input_prompt = ex["input_prompt"]
        if "qwen3" in model_type:
            input_ids = tokenizer.apply_chat_template(
                input_prompt, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=think)
        else:
            input_ids = tokenizer.apply_chat_template(
                input_prompt, tokenize=True, add_generation_prompt=True,
                return_tensors="pt")
    else:
        if use_chat_template:
            input_prompt = [{"role": "user", "content": ex["input_prompt"]}]
            if "qwen3" in model_type:
                input_ids = tokenizer.apply_chat_template(
                    input_prompt, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", enable_thinking=think)
            else:
                input_ids = tokenizer.apply_chat_template(
                    input_prompt, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt")
        else:
            input_ids = tokenizer.encode(ex["input_prompt"], return_tensors="pt")
    return input_ids


def setup_stop_token_ids(model, tokenizer, model_type):
    stop_token_ids = model.generation_config.eos_token_id
    stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
    stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
    stop_token_ids = list(set(
        [tokenizer.convert_tokens_to_ids(s) for s in stop] + stop_token_ids
    ))
    if "llama" in model_type:
        stop_token_ids.remove(tokenizer.unk_token_id)
    stop_token_ids = [x for x in stop_token_ids if x is not None]
    return stop_token_ids


def get_decoding_kwargs(args, model_type):
    if "qwen3" in model_type and args.think:
        return {
            "max_new_tokens": args.max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
        }
    else:
        return {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.temperature > 0,
        }


def run_flash_generation(args, model, tokenizer, dataset, model_type):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = None
    if args.stop_on_newline:
        stop_token_ids = setup_stop_token_ids(model, tokenizer, model_type)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline)

    outputs = []
    for i, ex in tqdm(enumerate(dataset), desc="Running generation", total=len(dataset)):
        if isinstance(ex["input_prompt"], list) and not args.use_chat_template:
            raise ValueError("Input prompt is already a list, chat template should be applied.")

        input_ids = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
        input_ids = input_ids.to(model.device)

        decoding_kwargs = get_decoding_kwargs(args, model_type)
        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        content = model.generate(input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken
        content = tokenizer.decode(content[0][input_ids.shape[1]:], skip_special_tokens=True)
        if i < 2:
            print("-" * 50)
            print("<INPUT PROMPT>")
            print(ex["input_prompt"])
            print("<OUTPUT>")
            print(content)
        outputs.append({"prompt": ex["input_prompt"], "output": content, "success": True, "time_taken": time_taken})
    return outputs


def run_rescale_generation(args, model, tokenizer, dataset, model_type,
                           rescale_config):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = None
    if args.stop_on_newline:
        stop_token_ids = setup_stop_token_ids(model, tokenizer, model_type)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline)
    print("Rescale config", rescale_config.__dict__)

    outputs = []
    for i, ex in tqdm(enumerate(dataset), desc="Running generation", total=len(dataset)):
        if isinstance(ex["input_prompt"], list) and not args.use_chat_template:
            raise ValueError("Input prompt is already a list, chat template should be applied.")

        input_ids = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
        print("INPUT LENGTH", input_ids.shape[1])

        input_ids = input_ids.to(model.device)

        decoding_kwargs = get_decoding_kwargs(args, model_type)
        decoding_kwargs["rescale_config"] = rescale_config
        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        content, generation_logging = model.rescale_generate(input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken
        content = tokenizer.decode(content[0][input_ids.shape[1]:], skip_special_tokens=True)
        if i < 2:
            print("-" * 50)
            print("<INPUT PROMPT>")
            print(ex["input_prompt"])
            print("<OUTPUT>")
            print(content)
        outputs.append({
            "prompt": ex["input_prompt"], "output": content, "success": True,
            "time_taken": time_taken, "generation_logging": generation_logging,
        })
    return outputs


def run_attnsharp_generation(args, model, tokenizer, dataset, model_type):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = None
    if args.stop_on_newline:
        stop_token_ids = setup_stop_token_ids(model, tokenizer, model_type)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline)

    outputs = []
    for i, ex in tqdm(enumerate(dataset), desc="Running generation", total=len(dataset)):
        if isinstance(ex["input_prompt"], list) and not args.use_chat_template:
            raise ValueError("Input prompt is already a list, chat template should be applied.")

        input_ids = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
        input_ids = input_ids.to(model.device)

        decoding_kwargs = get_decoding_kwargs(args, model_type)
        decoding_kwargs["use_attnsharp"] = True
        decoding_kwargs["attention_logits_temperature"] = args.attention_logits_temperature
        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        content, generation_logging = model.rescale_generate(input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken
        content = tokenizer.decode(content[0][input_ids.shape[1]:], skip_special_tokens=True)
        if i < 2:
            print("-" * 50)
            print("<INPUT PROMPT>")
            print(ex["input_prompt"])
            print("<OUTPUT>")
            print(content)
        outputs.append({
            "prompt": ex["input_prompt"], "output": content, "success": True,
            "time_taken": time_taken, "generation_logging": generation_logging,
        })
    return outputs


def main():
    args = _parse_args()

    # Validate
    if args.think and "qwen3" not in args.model.lower():
        raise ValueError("Thinking mode is only supported for Qwen3 models.")
    if args.decoding_method == "attnsharp":
        assert args.attention_logits_temperature is not None and args.attention_logits_temperature > 0

    # Detect model type
    model_type = detect_model_type(args.model)
    print(f"Detected model type: {model_type}")
    BaseModelClass, RescaleModelClass = _MODEL_CLASSES[model_type]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build rescale config if needed
    rescale_config = None

    if args.decoding_method == "dysco":
        rescale_config = build_rescale_config(args, tokenizer, model_type)

    # Get output path
    output_path = get_output_path(args, rescale_config)
    print("OUTPUT PATH", output_path)

    # Auto skip
    if args.auto_skip:
        score_path = output_path.replace(".json", "scores.json")
        if os.path.exists(output_path) and os.path.exists(score_path):
            print(f"Auto-skipping: both {output_path} and {score_path} exist")
            return

    # Load and filter data
    dataset, eval_func = load_eval_data(args.dataset)
    if not args.dataset.startswith("mrcr") and not args.dataset.startswith("graphwalks") and not args.dataset.startswith("clipper"):
        dataset = filter_dataset_by_length(args, dataset)
    print(f"Dataset size: {len(dataset)}")
    if args.test_size > 0:
        random.seed(args.seed)
        if not args.dataset.startswith("clipper"):
            random.shuffle(dataset)
        dataset = dataset[:args.test_size]

    # Load model
    if args.decoding_method == "flash":
        model = BaseModelClass.from_pretrained(
            args.model, device_map="auto", attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16)
    elif args.decoding_method in ["dysco", "attnsharp"]:
        model = RescaleModelClass.from_pretrained(
            args.model, attn_implementation="flash_attention_2", device_map="auto",
            torch_dtype=torch.bfloat16)
        print("Using model class:", RescaleModelClass.__name__)

    # Run generation
    if args.decoding_method == "flash":
        outputs = run_flash_generation(args, model, tokenizer, dataset, model_type)
    elif args.decoding_method == "dysco":
        outputs = run_rescale_generation(
            args, model, tokenizer, dataset, model_type,
            rescale_config)
    elif args.decoding_method == "attnsharp":
        outputs = run_attnsharp_generation(args, model, tokenizer, dataset, model_type)

    # Evaluate
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
            "output": {
                "prompt": output["prompt"], "output": output["output"],
                "success": output["success"], "thinking_part": thinking_part,
                "generation_logging": output.get("generation_logging", None),
                "time_taken": output["time_taken"],
            },
        }
        if args.skip_eval:
            ex_saving["metric"] = None
        else:
            mets, _ = eval_func(output["output"], ex)
            all_metrics.append(mets)
            ex_saving["metric"] = mets
        saving_info.append(ex_saving)

    # Save
    if args.skip_eval:
        output_content = {
            "args": args.__dict__,
            "saving_info": saving_info,
            "test_size": len(dataset),
        }
        with open(output_path, "w") as f:
            json.dump(output_content, f, indent=2)
        return

    # CLIPPER paired evaluation
    if "clipper" in args.dataset:
        assert len(all_metrics) % 2 == 0
        num_pairs = len(all_metrics) // 2
        paired_correct = sum(
            1 for i in range(num_pairs)
            if all_metrics[2*i]["accuracy"] == 1 and all_metrics[2*i+1]["accuracy"] == 1
        )
        avg_metrics = {"accuracy": paired_correct / num_pairs}
        for k in all_metrics[0].keys():
            if k != "accuracy":
                avg_metrics[k] = np.mean([x[k] for x in all_metrics])
    else:
        avg_metrics = {k: np.mean([x[k] for x in all_metrics]) for k in all_metrics[0].keys()}

    print([f"{k}: {v*100:.1f}" for k, v in avg_metrics.items()])

    output_content = {
        "args": args.__dict__,
        "saving_info": saving_info,
        "avg_metrics": avg_metrics,
        "test_size": len(dataset),
    }
    with open(output_path, "w") as f:
        json.dump(output_content, f, indent=2)
    with open(output_path.replace(".json", "scores.json"), "w") as f:
        json.dump(avg_metrics, f, indent=2)


if __name__ == "__main__":
    main()
