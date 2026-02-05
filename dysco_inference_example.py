"""Minimal example: run DySCO with Qwen3-8B on one path_walking_16k example."""

import json
import yaml
import torch
from transformers import AutoTokenizer
from dysco.custom_modeling_qwen3 import RescaleQwen3ForCausalLM
from dysco.custom_mixin import DyscoConfig

# ---------- paths (adjust as needed) ----------
MODEL_PATH = "models/Qwen3-8B"
DYSCO_CFG_PATH = "dysco_cfgs/qwen3_8b.yaml"

# ---------- 1. load one example ----------
DATA_DIR = "data_eval/longproc/path_walking"
with open(f"{DATA_DIR}/path_walking_16k.json") as f:
    raw = json.load(f)[0]

with open(f"{DATA_DIR}/prompts.yaml") as f:
    user_prompt_template = yaml.safe_load(f)["USER_PROMPT"]

prompt = user_prompt_template.format(
    city_context=raw["context_nl"],
    src_city=raw["question_repr"][0],
    dst_city=raw["question_repr"][1],
)

# ---------- 2. load model & tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = RescaleQwen3ForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="flash_attention_2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ---------- 3. build DyscoConfig from yaml ----------
with open(DYSCO_CFG_PATH) as f:
    cfg = yaml.safe_load(f)

# parse qrheads string -> list of tuples
if isinstance(cfg["dysco_qrheads"], str):
    cfg["dysco_qrheads"] = eval(cfg["dysco_qrheads"])

# resolve 'auto' intervention warmup: number of generation-prompt tokens
if cfg["dysco_interv_warmup"] == "auto":
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
    )
    dummy_no_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=True, add_generation_prompt=False, return_tensors="pt", enable_thinking=False,
    )
    cfg["dysco_interv_warmup"] = dummy.shape[1] - dummy_no_gen.shape[1]

dysco_config = DyscoConfig(
    dysco_qrheads=cfg["dysco_qrheads"],
    dysco_top_k=cfg["dysco_top_k"],
    dysco_top_p=cfg["dysco_top_p"],
    dysco_strength=cfg["dysco_strength"],
    dysco_ctx_momentum=cfg["dysco_ctx_momentum"],
    dysco_ctx_warmup=cfg["dysco_ctx_warmup"],
    dysco_interv_warmup=cfg["dysco_interv_warmup"],
    dysco_template_seqs=cfg.get("dysco_template_seqs"),
)

# ---------- 4. tokenize ----------
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", enable_thinking=False,
).to(model.device)
input_len = input_ids.shape[1]
print(f"Input length: {input_len} tokens")

# ---------- 5. generate with DySCO ----------
generated_ids, generation_logging = model.dysco_generate(
    input_ids,
    dysco_config=dysco_config,
    max_new_tokens=512,
    temperature=0.0,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)

output = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

# ---------- 6. print results ----------
print(f"\nSource: {raw['question_repr'][0]}")
print(f"Destination: {raw['question_repr'][1]}")
print(f"\n--- Model output ---\n{output}")
