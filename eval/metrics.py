import logging
from typing import Any, Dict, List

import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Calculates ROUGE and BLEU scores for a list of predictions and references.

    Args:
        predictions (List[str]): The list of predicted texts.
        references (List[str]): The list of reference texts.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated scores.
    """
    logging.info("Calculating ROUGE and BLEU scores.")
    try:
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")

        rouge_results = rouge.compute(predictions=predictions, references=references)
        bleu_results = bleu.compute(predictions=predictions, references=references)

        results = {**rouge_results, **bleu_results}  # type: ignore
        logging.info(f"Calculated metrics: {results}")
        return results
    except Exception as e:
        logging.error(f"Failed to calculate metrics: {e}")
        return {}
