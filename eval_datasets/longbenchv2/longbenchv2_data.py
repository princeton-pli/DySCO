import os
import json
import re


##### ADAPTED from original Longbenchv2 #####
_PROMPT_TEMPLATE_DIRECT = """
Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert choice here)".
""".strip()


_PROMPT_TEMPLATE_ORIGINAL_COT = """
Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step. After thinking, choose a single, most likely answer. Output your final answer follows: "The correct answer is (insert choice here)".
""".strip()


_PROMPT_TEMPLATE_COT = """
Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step first and then choose a single answer. Format your response following the example below.

Example Response Format:
REASONING:
(reasoning process here)

ANSWER:
The correct answer is (insert choice here).
""".strip()



def _clean_data():
    """
    Clean the original longbenchv2 data to fit into different context sizes.
    """

    orig_data = "data_eval/lbv2/data.json"
    with open(orig_data, "r") as f:
        data = json.load(f)
    print(len(data))

    from transformers import AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained("models/Llama-2-7b-chat-hf")
    len_input = []
    kept_data = []
    _max_threshold = (1024 * 64) - 256 # 256 as buffer

    for d in tqdm(data):
        context = d["context"]
        input_prompt = _PROMPT_TEMPLATE.replace('$DOC$', context.strip()).replace('$Q$', d['question'].strip()).replace('$C_A$', d['choice_A'].strip()).replace('$C_B$', d['choice_B'].strip()).replace('$C_C$', d['choice_C'].strip()).replace('$C_D$', d['choice_D'].strip())
        num_tokens = len(tokenizer(input_prompt)["input_ids"])

        if num_tokens < _max_threshold:
            kept_data.append(d)
        len_input.append(num_tokens)

    print(max(len_input))
    print(len(kept_data))
    with open(f"data_eval/lbv2/data_{64}k.json", "w") as f:
        json.dump(kept_data, f)


def _clean_data_qr():
    """
    Clean the original longbenchv2 data to fit into different context sizes.
    """

    orig_data = "data_eval/lbv2_qr/data.json"
    with open(orig_data, "r") as f:
        data = json.load(f)
    print(len(data))

    from transformers import AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained("models/Llama-2-7b-chat-hf")
    len_input = []
    kept_data = []
    _min_threshold = (1024 * 128) - 256 # 256 as buffer
    _max_threshold = (1024 * 256) - 256 # 256 as buffer

    for d in tqdm(data):
        context = d["context"]
        input_prompt = _PROMPT_TEMPLATE_DIRECT.replace('$DOC$', context.strip()).replace('$Q$', d['question'].strip()).replace('$C_A$', d['choice_A'].strip()).replace('$C_B$', d['choice_B'].strip()).replace('$C_C$', d['choice_C'].strip()).replace('$C_D$', d['choice_D'].strip())
        num_tokens = len(tokenizer(input_prompt)["input_ids"])

        if _min_threshold <= num_tokens < _max_threshold:
            kept_data.append(d)
        len_input.append(num_tokens)

    print(max(len_input))
    print(len(kept_data))
    with open(f"data_eval/lbv2_qr/data_{256}k.json", "w") as f:
        json.dump(kept_data, f)



def _extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def _eval_longbenchv2(prediction: str, example: dict):
    gt = example["item"]["answer"].strip()
    parsed_output = _extract_answer(prediction)
    if parsed_output is None:
        return (
            {
                "accuracy": 0.0,
                "extraction": 0.0,
            },
            {
                "parsed_output": None,
            },
        )

    accuracy = 1.0 if gt == parsed_output else 0.0
    extraction = 1.0
    return (
        {
            "accuracy": accuracy,
            "extraction": extraction
        },
        {
            "parsed_output": parsed_output
        }
    )

def load_longbenchv2_data(dataset_name: str, path: str = "data_eval/lbv2_qr"):
    # assert dataset_name in ["longbenchv2_cot_32k", "longbenchv2_cot_64k","longbenchv2_cot_128k", "longbenchv2_32k", "longbenchv2_64k", "longbenchv2_128k", "longbenchv2_128k_sp1", "longbenchv2_128k_sp2"]
    if dataset_name == "longbenchv2_128k_sp1":
        data_file = "data_128k_sp1.json"
    elif dataset_name == "longbenchv2_128k_sp2":
        data_file = "data_128k_sp2.json"
    else:
        data_file = f"data_{dataset_name.split('_')[-1]}.json"
    with open(os.path.join(path, data_file), "r") as f:
        data = json.load(f)

    if dataset_name.startswith("longbenchv2_cot"):
        prompt_template = _PROMPT_TEMPLATE_COT
    elif dataset_name.startswith("longbenchv2_original_cot"):
        prompt_template = _PROMPT_TEMPLATE_ORIGINAL_COT
    else:
        prompt_template = _PROMPT_TEMPLATE_DIRECT
    # keys(['_id', 'domain', 'sub_domain', 'difficulty', 'length', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer', 'context'])
    purged_data = []
    for d in data:
        context = d["context"]
        input_prompt = prompt_template.replace('$DOC$', context.strip()).replace('$Q$', d['question'].strip()).replace('$C_A$', d['choice_A'].strip()).replace('$C_B$', d['choice_B'].strip()).replace('$C_C$', d['choice_C'].strip()).replace('$C_D$', d['choice_D'].strip())
        reference_output = f"The correct answer is ({d['answer']})"
        del d["context"]
        del d["choice_A"]
        del d["choice_B"]
        del d["choice_C"]
        del d["choice_D"]
        purged_data.append({
            "input_prompt": input_prompt,
            "reference_output": reference_output,
            "item": d
        })
        
    print(f"Loaded {len(purged_data)} datapoints for {dataset_name} from {data_file}")
    return purged_data, _eval_longbenchv2




def load_retrieved_longbenchv2_data(dataset_name: str, data_path: str, retrieval_path: str, retriever: str, k: int):
    """
    Load LongBenchV2 data with retrieved paragraphs.

    Args:
        dataset_name: e.g., "longbenchv2_32k", "longbenchv2_cot_64k"
        data_path: path to chunked data files (e.g., "data_eval/lbv2")
        retrieval_path: path to retrieval results (e.g., "results_retriever/lbv2")
        retriever: retriever name (e.g., "stella")
        k: top-k documents to retrieve
    """
    # Extract context length suffix (e.g., "32k", "64k", "128k")
    context_len = dataset_name.split('_')[-1]

    # Load chunked data
    data_file = f"data_{context_len}_chunk1024.json"
    with open(os.path.join(data_path, data_file), "r") as f:
        data = json.load(f)

    # Load retrieval results
    retrieval_file = f"data_{context_len}_chunk1024_flat-{retriever}.json"
    with open(os.path.join(retrieval_path, retrieval_file), "r") as f:
        retrieval = json.load(f)

    # Select prompt template
    if dataset_name.startswith("longbenchv2_cot"):
        prompt_template = _PROMPT_TEMPLATE_COT
    elif dataset_name.startswith("longbenchv2_original_cot"):
        prompt_template = _PROMPT_TEMPLATE_ORIGINAL_COT
    else:
        prompt_template = _PROMPT_TEMPLATE_DIRECT

    def prepare_sample(sample, retrieval_dict):
        """
        Prepare a single LongBenchV2 sample using retrieved paragraphs.
        """
        # Get top-k paragraph indices from retrieval, sorted by score descending
        top_k_keys = sorted(retrieval_dict.keys(), key=lambda x: retrieval_dict[x], reverse=True)[:k]

        # Sort by paragraph index (ascending) to preserve document order
        result_doc_idx = sorted(top_k_keys, key=lambda x: int(x))

        # Build id to text mapping from sample's paragraphs
        id2text = {str(doc['idx']): doc['paragraph_text'] for doc in sample["paragraphs"]}

        # Get texts for selected docs
        result_docs = [id2text[idx] for idx in result_doc_idx]

        # Concatenate context
        context = "\n\n\n".join(result_docs)

        # Prepare prompt
        input_prompt = prompt_template.replace('$DOC$', context.strip()).replace('$Q$', sample['question'].strip()).replace('$C_A$', sample['choice_A'].strip()).replace('$C_B$', sample['choice_B'].strip()).replace('$C_C$', sample['choice_C'].strip()).replace('$C_D$', sample['choice_D'].strip())

        reference_output = f"The correct answer is ({sample['answer']})"

        # Remove fields we don't need in item (to save memory)
        item = {k: v for k, v in sample.items() if k not in ["paragraphs", "choice_A", "choice_B", "choice_C", "choice_D"]}

        return {
            "input_prompt": input_prompt,
            "reference_output": reference_output,
            "item": item
        }

    # Prepare samples
    purged_data = []
    for sample in data:
        sample_id = sample["_id"]

        purged_data.append(prepare_sample(sample, retrieval[sample_id]))

    print(f"Loaded {len(purged_data)} datapoints for {dataset_name} with retriever {retriever} (k={k})")
    return purged_data, _eval_longbenchv2



if __name__ == "__main__":
    # _clean_data()
    _clean_data_qr()