import os
import json

from .clipper_utils import evaluate_clipper_single, parse_clipper_answer



def post_process(output, example):
    """
    Evaluate CLIPPER output against expected label.

    CLIPPER uses paired evaluation: each TRUE sample is paired with its corresponding
    FALSE sample (by position). A pair is only correct when TRUE answer is "true"
    AND FALSE answer is "false".

    Individual sample metrics are still computed for analysis.

    Args:
        output: Model generated text
        example: Original example dict with 'item' key

    Returns:
        metrics: Dict with accuracy and parsing success metrics
        details: Dict with parsed and expected answers for debugging
    """
    example_item = example["item"]

    # Get expected answer
    expected_label = example_item["label"]

    # Evaluate individual sample (for debugging/analysis)
    metrics = evaluate_clipper_single(output, expected_label)

    parsed_answer = parse_clipper_answer(output)
    expected_answer = "true" if expected_label else "false"

    details = {
        "parsed_answer": parsed_answer,
        "expected_answer": expected_answer,
        "qid": example_item.get("idx", "unknown"),
        "is_true_sample": expected_label,  # Track if this is from TRUE or FALSE file
    }

    return metrics, details




def load_clipper_data(dataset_name: str, path: str):
    """
    Load CLIPPER dataset for TRUE/FALSE classification based on context.

    CLIPPER evaluates models on their ability to verify statements against long contexts.
    The dataset consists of TWO files:
    - One file with TRUE statements (test_data_TRUE.json)
    - One file with FALSE statements (test_data_FALSE.json)

    Both files need to be evaluated together for full accuracy metrics.

    Expected input data format for each file (JSON):
    [
        {
            "idx": "unique_id",
            "question": "statement to verify",
            "paragraphs": [{"idx": 0, "paragraph_text": "content"}, ...],
            "gt_docs": ["0", "1", ...],  # ground truth relevant doc indices (not used in baseline)
            "num_gold_docs": int,
        },
        ...
    ]

    Dataset naming: clipper (test set)
    The suffix indicates approximate context length.
    """

    assert dataset_name in ["clipper", "clipper_sp1", "clipper_sp2"]

    # Load pre-subsampled data: [{"data": sample, "label": True/False}, ...]
    if dataset_name == "clipper_sp1":
        data_path = os.path.join(path, "test-400_sp1.json")
    elif dataset_name == "clipper_sp2":
        data_path = os.path.join(path, "test-400_sp2.json")
    else:
        data_path = os.path.join(path, "test-400.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generation instruction
    generation_instruction = "You are provided with a context and a statement. Your task is to carefully read the context and then determine whether the statement is true or false.  \n\nAnswer TRUE if the statement is true in its entirety based on the context provided.\nAnswer FALSE if any part of the statement is false based on the context provided.\n\n<context>{}</context>\n\n\n<statement>{}</statement>\n\n<question>Based on the context provided, is the above statement TRUE or FALSE?</question>\n\nFirst provide an explanation of your decision-making process, and then provide your final answer. Use the following format:\n\n<explanation>YOUR EXPLANATION</explanation>\n<answer>YOUR ANSWER</answer>"

    data_purged = []
    for item in data:
        sample = item["data"]
        expected_label = item["label"]

        docs = sample["paragraphs"]
        result_doc_idx = sorted([str(doc['idx']) for doc in docs], key=lambda x: int(x))
        id2text = {str(doc['idx']): doc['paragraph_text'] for doc in docs}
        result_docs = [id2text[idx] for idx in result_doc_idx]
        context = "\n\n\n".join(result_docs)
        input_prompt = generation_instruction.format(context, sample["question"])
        reference_output = "TRUE" if expected_label else "FALSE"
        sample_with_label = {**sample, "label": expected_label}

        data_purged.append({
            "input_prompt": input_prompt,
            "reference_output": reference_output,
            "item": sample_with_label
        })

    return data_purged, post_process








def load_retrieved_clipper_data(dataset_name: str, data_path: str, retrieval_path: str, retriever: str, k: int):
    """
    return list of datapoints ({"input_prompt", "reference_output", "item" }) and a eval func;
    only support clipper
    """

    assert dataset_name in ["clipper"]

    true_data_path = os.path.join(data_path, "test-00000-of-00002_chunk1024.json")
    false_data_path = os.path.join(data_path, "test-00001-of-00002_chunk1024.json")

    true_retrieval_path = os.path.join(retrieval_path, f"test-00000-of-00002_chunk1024_flat-{retriever}.json")
    false_retrieval_path = os.path.join(retrieval_path, f"test-00001-of-00002_chunk1024_flat-{retriever}.json")

    # Load data from JSON files
    with open(true_data_path, "r", encoding="utf-8") as f:
        true_data = json.load(f)
    with open(false_data_path, "r", encoding="utf-8") as f:
        false_data = json.load(f)

    with open(true_retrieval_path, "r", encoding="utf-8") as f:
        true_retrieval = json.load(f)
    with open(false_retrieval_path, "r", encoding="utf-8") as f:
        false_retrieval = json.load(f)

    # Generation instruction
    generation_instruction = "You are provided with a context and a statement. Your task is to carefully read the context and then determine whether the statement is true or false.  \n\nAnswer TRUE if the statement is true in its entirety based on the context provided.\nAnswer FALSE if any part of the statement is false based on the context provided.\n\n<context>{}</context>\n\n\n<statement>{}</statement>\n\n<question>Based on the context provided, is the above statement TRUE or FALSE?</question>\n\nFirst provide an explanation of your decision-making process, and then provide your final answer. Use the following format:\n\n<explanation>YOUR EXPLANATION</explanation>\n<answer>YOUR ANSWER</answer>"

    def prepare_sample(sample, retrieval_dict, expected_label):
        """
        Prepare a single CLIPPER sample for evaluation using retrieved paragraphs.

        Args:
            sample: Raw data sample with paragraphs
            retrieval_dict: Dict of {paragraph_idx: score} sorted by score descending
            expected_label: True for TRUE statements, False for FALSE statements
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
        input_prompt = generation_instruction.format(context, sample["question"])

        # Set expected answer
        reference_output = "TRUE" if expected_label else "FALSE"

        # Store expected label in item for evaluation
        sample_with_label = {**sample, "label": expected_label}

        return {
            "input_prompt": input_prompt,
            "reference_output": reference_output,
            "item": sample_with_label
        }

    # Prepare samples INTERLEAVED as pairs: (TRUE[0], FALSE[0], TRUE[1], FALSE[1], ...)
    data_purged = []
    for true_sample, false_sample in zip(true_data, false_data):
        true_idx = str(true_sample["idx"])
        false_idx = str(false_sample["idx"])

        # Skip if retrieval results not available for either sample
        if true_idx not in true_retrieval or false_idx not in false_retrieval:
            continue

        # Add TRUE sample with its retrieval
        data_purged.append(prepare_sample(true_sample, true_retrieval[true_idx], expected_label=True))
        # Add corresponding FALSE sample with its retrieval
        data_purged.append(prepare_sample(false_sample, false_retrieval[false_idx], expected_label=False))

    # Verify interleaved structure: (TRUE[0], FALSE[0], TRUE[1], FALSE[1], ...)
    # Even indices (0, 2, 4, ...) should be TRUE samples
    # Odd indices (1, 3, 5, ...) should be FALSE samples with idx = TRUE idx + 1000
    for i in range(0, len(data_purged), 2):
        true_item = data_purged[i]["item"]
        false_item = data_purged[i + 1]["item"]
        assert true_item["label"] == True, f"Position {i} should be TRUE sample"
        assert false_item["label"] == False, f"Position {i + 1} should be FALSE sample"
        assert int(false_item["idx"]) == int(true_item["idx"]) + 1000, (
            f"Pair mismatch at position {i}: true_idx={true_item['idx']}, false_idx={false_item['idx']} "
            f"(expected false_idx={int(true_item['idx']) + 1000})"
        )

    return data_purged, post_process
