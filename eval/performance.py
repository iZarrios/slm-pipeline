import logging
import time
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def measure_performance(model: Any, tokenizer: Any, queries: List[str]) -> Dict[str, float]:
    """
    Measures the inference latency and throughput of a model.

    Args:
        model (Any): The model to evaluate.
        tokenizer (Any): The tokenizer for the model.
        queries (List[str]): A list of queries to test the model with.

    Returns:
        Dict[str, float]: A dictionary containing the latency and throughput.
    """
    logging.info("Measuring model performance.")
    total_time = 0
    num_queries = len(queries)

    for query in queries:
        start_time = time.time()
        inputs = tokenizer(query, return_tensors="pt")
        _ = model.generate(**inputs, max_length=50)
        end_time = time.time()
        total_time += end_time - start_time

    latency = total_time / num_queries
    throughput = num_queries / total_time

    performance_metrics = {
        "latency_seconds": latency,
        "throughput_queries_per_second": throughput,
    }
    logging.info(f"Measured performance: {performance_metrics}")
    return performance_metrics
