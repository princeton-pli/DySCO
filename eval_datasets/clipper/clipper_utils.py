import re


def parse_clipper_answer(output: str) -> str:
    """
    Parse the answer from CLIPPER model output.

    Expected format: <answer>TRUE/FALSE</answer>

    Args:
        output: Model output string

    Returns:
        Parsed answer as lowercase string ('true', 'false', or None if unparseable)
    """
    try:
        tag = "answer"
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
        match = pattern.findall(output)
        if match:
            return match[0].strip().lower()
    except:
        pass

    # Fallback: search for true/false in output
    output_lower = output.lower()

    # Check for explicit true/false statements
    if "true" in output_lower and "false" not in output_lower:
        return "true"
    elif "false" in output_lower and "true" not in output_lower:
        return "false"

    return None



def evaluate_clipper_single(output: str, expected_label: bool) -> dict:
    """
    Evaluate a single CLIPPER output.

    Args:
        output: Model output string
        expected_label: True if statement should be TRUE, False if should be FALSE

    Returns:
        Dictionary with accuracy metrics
    """
    parsed_answer = parse_clipper_answer(output)
    expected_answer = "true" if expected_label else "false"

    correct = (parsed_answer == expected_answer)
    parsed_successfully = (parsed_answer is not None)

    return {
        "accuracy": 1 if correct else 0,
        "parsed_successfully": 1 if parsed_successfully else 0,
    }
