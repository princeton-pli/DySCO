"""Minimal example: run DySCO with Qwen3-8B on one path_walking_16k example."""

import json
import yaml
import torch
from transformers import AutoTokenizer
from dysco.custom_modeling_qwen3 import RescaleQwen3ForCausalLM
from dysco.custom_mixin import RescaleConfig

# ---------- paths (adjust as needed) ----------
MODEL_PATH = "models/Qwen3-8B"
DYSCO_CFG_PATH = "dysco_cfgs/qwen3_8b.yaml"

# ---------- 1. load one example ----------
DATA_DIR = "data_eval/longproc/path_walking"
with open(f"{DATA_DIR}/path_walking_16k.json") as f:
    raw = json.load(f)[187]  # Oakland -> Royston (a known successful example)

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

# ---------- 3. build RescaleConfig from yaml ----------
with open(DYSCO_CFG_PATH) as f:
    cfg = yaml.safe_load(f)

# parse selected_heads string -> list of tuples
selected_heads = cfg["selected_heads"]
if isinstance(selected_heads, str):
    selected_heads = eval(selected_heads)

# resolve 'auto' intervention warmup: number of generation-prompt tokens
intervention_warmup = cfg.get("intervention_warmup", "auto")
if intervention_warmup == "auto":
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
    )
    dummy_no_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=True, add_generation_prompt=False, return_tensors="pt", enable_thinking=False,
    )
    intervention_warmup = dummy.shape[1] - dummy_no_gen.shape[1]

# build template_sequences list
template_sequences = cfg.get("template_sequences", [])
template_sequences = [torch.LongTensor(seq) for seq in template_sequences]

rescale_config = RescaleConfig(
    selected_heads=selected_heads,
    top_k=1024,           # override: use top-k only for this quick demo
    top_p=None,
    strength=2.5,
    decay_factor=cfg["decay_factor"],
    context_warmup_steps=cfg.get("context_warmup_steps", 8),
    intervention_warmup_steps=intervention_warmup,
    dynamic_rescale=cfg.get("dynamic_rescale", True),
    template_sequences=template_sequences,
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
generated_ids, generation_logging = model.rescale_generate(
    input_ids,
    rescale_config=rescale_config,
    max_new_tokens=128,
    temperature=0.0,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)

output = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

# ---------- 6. print results ----------
print(f"\nSource: {raw['question_repr'][0]}")
print(f"Destination: {raw['question_repr'][1]}")
print(f"\n--- Model output ---\n{output}")
# Expected output (Oakland -> Royston):
#   <Route>
#   From Oakland, take a train to Tours.
#   From Tours, take a plane to Las Vegas.
#   From Las Vegas, take a plane to Salem.
#   From Salem, take a bus to Royston.
#   </Route>
