import logging

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.metrics import calculate_metrics
from eval.performance import measure_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_model(
    model_path: str,
    baseline_model_path: str,
    test_data_path: str,
):
    """
    Evaluates a fine-tuned model against a baseline model.

    Args:
        model_path (str): The path to the fine-tuned model.
        baseline_model_path (str): The path to the baseline model.
        test_data_path (str): The path to the test data.
    """
    logging.info(f"Starting evaluation of model: {model_path}")
    logging.info(f"Baseline model: {baseline_model_path}")
    logging.info(f"Test data: {test_data_path}")

    # Load models and tokenizers
    logging.info("Loading models and tokenizers.")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_model_path)
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)

    # Load test data
    logging.info("Loading test data.")
    test_df = pd.read_json(test_data_path, lines=True)
    queries = test_df["question"].tolist()
    references = test_df["answer"].tolist()

    # Generate predictions
    logging.info("Generating predictions from fine-tuned model.")
    predictions = []
    for query in queries:
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(response)

    logging.info("Generating predictions from baseline model.")
    baseline_predictions = []
    for query in queries:
        inputs = baseline_tokenizer(query, return_tensers="pt")
        outputs = baseline_model.generate(**inputs, max_length=50)
        response = baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_predictions.append(response)

    # Calculate metrics
    logging.info("Calculating metrics for fine-tuned model.")
    metrics = calculate_metrics(predictions, references)

    logging.info("Calculating metrics for baseline model.")
    baseline_metrics = calculate_metrics(baseline_predictions, references)

    # Measure performance
    logging.info("Measuring performance of fine-tuned model.")
    performance = measure_performance(model, tokenizer, queries)

    logging.info("Measuring performance of baseline model.")
    baseline_performance = measure_performance(
        baseline_model, baseline_tokenizer, queries
    )

    # Print report
    print("\n--- Evaluation Report ---")
    print("\n--- Fine-tuned Model ---")
    print("Metrics:", metrics)
    print("Performance:", performance)
    print("\n--- Baseline Model ---")
    print("Metrics:", baseline_metrics)
    print("Performance:", baseline_performance)
    print("\n--- End of Report ---")


if __name__ == "__main__":
    # This is an example of how to run the evaluation.
    # You would need to replace the paths with your actual model and data paths.
    evaluate_model(
        model_path="models/fine-tuned-model",
        baseline_model_path="gpt2",  # Example baseline
        test_data_path="data/test.json",
    )
