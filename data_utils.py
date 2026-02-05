from eval_datasets.longproc.longproc_data import load_longproc_data
from eval_datasets.longbenchv2.longbenchv2_data import load_longbenchv2_data
from eval_datasets.helmet.helmet_data import load_helmet_data
from eval_datasets.mrcr.mrcr_data import load_mrcr_data
from eval_datasets.clipper.clipper_data import load_clipper_data

from eval_datasets.longbenchv2.longbenchv2_data import load_retrieved_longbenchv2_data
from eval_datasets.clipper.clipper_data import load_retrieved_clipper_data


_HELMET_DATASETS = [
    "hotpot",
    "infbench",
]

_LONGPROC_EVAL_DATASETS = [
    "path_walking",
]

_MRCR_EVAL_DATASETS = [
    "mrcr_2needle",
    "mrcr_4needle",
    "mrcr_8needle",
]

_LONGBENCHV2_EVAL_DATASETS = [
    "longbenchv2_"
]

_GRAPHWALKS_EVAL_DATASETS = [
    "graphwalks",
]

_CLIPPER_DATASETS = [
    "clipper"
]



def load_retrieved_eval_data(dataset_name: str, retriever: str, k: int):
    """
    return list of datapoints ({"input_prompt", "reference_output", "item" }) and a eval func;
    only support clipper and longbenchv2
    """

    if any([dataset_name.startswith(prefix) for prefix in _LONGBENCHV2_EVAL_DATASETS]):
        return load_retrieved_longbenchv2_data(dataset_name, data_path="data_eval/lbv2_qr", retrieval_path="results_retriever/lbv2", retriever=retriever, k=k)
    elif any([dataset_name.startswith(prefix) for prefix in _CLIPPER_DATASETS]):
        return load_retrieved_clipper_data(dataset_name, data_path="data_eval/clipper", retrieval_path="results_retriever/clipper", retriever=retriever, k=k)
    else:
        raise ValueError(f"Unknown retriever eval dataset: {dataset_name}")



def load_eval_data(dataset_name: str):
    """
    return list of datapoints ({"input_prompt", "reference_output", "item" }) and a eval func;
    """

    if any([dataset_name.startswith(prefix) for prefix in _LONGPROC_EVAL_DATASETS]):
        return load_longproc_data(dataset_name, path="data_eval/longproc",)
    elif any([dataset_name.startswith(prefix) for prefix in _LONGBENCHV2_EVAL_DATASETS]):
        return load_longbenchv2_data(dataset_name, path="data_eval/lbv2_qr")
    elif any([dataset_name.startswith(prefix) for prefix in _HELMET_DATASETS]):
        return load_helmet_data(dataset_name, path="data_eval/helmet")
    elif any([dataset_name.startswith(prefix) for prefix in _MRCR_EVAL_DATASETS]):
        return load_mrcr_data(dataset_name, path="data_eval/mrcr")
    elif any([dataset_name.startswith(prefix) for prefix in _GRAPHWALKS_EVAL_DATASETS]):
        return load_graphwalks_data(dataset_name, path="data_eval/graphwalks")
    elif any([dataset_name.startswith(prefix) for prefix in _CLIPPER_DATASETS]):
        return load_clipper_data(dataset_name, path="data_eval/clipper")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def test_load_eval_data():
    dataset_name = "path_walking_16k"

    data, eval_func = load_eval_data(dataset_name)
    print(len(data))
    print(f"Loaded {len(data)} datapoints for {dataset_name}")
    print("*" * 50)
    print("PROMPT:")
    print(data[0]["input_prompt"] + "|||")
    print("*" * 50)
    print("REFERENCE OUTPUT:")
    print(data[0]["reference_output"])
    print("*" * 50)
    print("EVAL:")
    print(eval_func("<Route>" + data[0]["reference_output"] + "</Route>", data[0]))
    print("*" * 50)

if __name__ == "__main__":
    test_load_eval_data()