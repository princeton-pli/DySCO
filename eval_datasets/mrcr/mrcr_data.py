import json
from difflib import SequenceMatcher
import os


def _eval_mrcr(response, example) -> float:
    """
    Compare response and answer.
    """
    # remove spaces surrounding the response
    response = response.strip()
    answer = example["reference_output"]
    random_string_to_prepend = example["random_string_to_prepend"]

    if not response.startswith(random_string_to_prepend):
        return (
            {
                "match_ratio": 0.0
            },
            {
                "response": response
            }
        )
    
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    match_ratio = float(SequenceMatcher(None, response, answer).ratio())
    return (
        {
            "match_ratio": match_ratio
        },
        {
            "response": response
        }
    )



def load_mrcr_data(dataset_name: str, path: str = "data_eval/mrcr"):
    assert dataset_name in [
        "mrcr_2needle_8k",
        "mrcr_2needle_16k",
        "mrcr_2needle_32k",
        "mrcr_2needle_64k",
        "mrcr_2needle_128k",
        "mrcr_2needle_256k",
        "mrcr_4needle_8k",
        "mrcr_4needle_16k",
        "mrcr_4needle_32k",
        "mrcr_4needle_64k",
        "mrcr_4needle_128k",
        "mrcr_4needle_256k",
        "mrcr_8needle_8k",
        "mrcr_8needle_16k",
        "mrcr_8needle_32k",
        "mrcr_8needle_64k",
        "mrcr_8needle_128k",
        "mrcr_8needle_256k"
    ]

    first_split = dataset_name.split("_")
    needle_num = first_split[1]  # e.g., "2needle"
    length_kb = first_split[2]  # e.g., "16k"
    data_file = f"{needle_num}_{length_kb}.json"

    # Load data
    with open(os.path.join(path, data_file), "r") as f:
        purged_data = json.load(f)

    print(f"Loaded {len(purged_data)} examples from {dataset_name}")
    return purged_data, _eval_mrcr
