from typing import Tuple, List, Dict, Callable

import json
import os
import yaml
import re

def _extract_with_tag(response: str, tag: str):
    start = response.find(f"<{tag}>")
    end = response.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return response[start+len(tag)+2:end].strip()


def eval_path_traversal(prediction: str, example: dict):
    """
    Returns: metrics (dict) and additional info
    """
    # metric: accuracy, partial_accuracy
    gt = example["reference_output"]
    parsed_pred = _extract_with_tag(prediction, "Route")
    if parsed_pred is None:
        return {"accuracy": .0, "partial_accuracy": .0, "extraction_rate": .0}, {"parsed_output": None, "error_report": "Parsing error"}

    gt = gt.strip()
    parsed_pred = parsed_pred.strip()
    if gt == parsed_pred:
        return {"accuracy": 1.0, "partial_accuracy": 1.0, "extraction_rate": 1.0}, {"parsed_output": parsed_pred, "error_report": None}

    gt_lines = gt.split("\n")
    pred_lines = parsed_pred.split("\n")

    error_report = None
    for i, (gl, pl) in enumerate(zip(gt_lines, pred_lines)):
        if gl != pl:
            error_report = {"line": i, "gt": gl, "pr": pl}
            break
    i += 1
    return {"accuracy": 0.0, "partial_accuracy": i/len(gt_lines), "extraction_rate": 1.0}, {"parsed_output": parsed_pred, "error_report": error_report}


def _load_path_walking_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    # path is a dataset folder, containing different levels of path traversal data
    assert dataset_name in ["path_walking_4k", "path_walking_8k", "path_walking_16k", "path_walking_32k"]

    if path is None: path = "longproc_data"

    path = os.path.join(path, "path_walking")

    data_file = os.path.join(path, dataset_name + ".json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    data_purged = []
    for d in data:
        city_context = d["context_nl"]
        src_city = d["question_repr"][0]
        dst_city = d["question_repr"][1]
        data_purged.append({
            "city_context": city_context,
            "src_city": src_city,
            "dst_city": dst_city,
            "reference_output": d["answer_nl"],
        })

    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_path_traversal

def load_longproc_data(dataset_name: str, path: str=None) -> Tuple[List, Callable]:
    """
    Load the dataset and evaluation function given the dataset name and path.
    returns: list of data, evaluation function
    the data list will contain {"input_prompt", "reference_output", and "item"} for each data point
    """

    dataset_basename = dataset_name.rsplit("_", 1)[0]

    dataset_loaders = {
        "path_walking": _load_path_walking_data,
    }
    
    if dataset_basename in dataset_loaders:
        dataset_loading_func = dataset_loaders[dataset_basename]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    packed_data, eval_func = dataset_loading_func(dataset_name, path)

    template = packed_data["prompt_template"]
    data = packed_data["data"]

    unpacked_data = []
    for d in data:
        unpacked_data.append({
            "input_prompt": template.format(**d),
            "reference_output": d["reference_output"],
            "item": d
        })

    assert all(["input_prompt" in d and "reference_output" in d and "item" in d for d in unpacked_data])

    return unpacked_data, eval_func

