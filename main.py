import json
import logging
import os
import subprocess
import sys

import openai
import requests
import yaml

from data_collection.pdf_processor import extract_pdf_with_layout
from data_collection.web_scraper import scrape_with_attribution
from data_processing.chunker import chunk_text
from data_processing.generation_without_context import get_openai_completion
from data_processing.setup_dataset import setup_dataset
from training.train import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def deploy_model():
    """
    Deploys the model using FastAPI.
    """
    logging.info("Deploying model...")
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "deploy:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
        )
        logging.info("FastAPI server started at http://0.0.0.0:8000")
    except FileNotFoundError:
        logging.error("Could not find 'uvicorn'. Please install it with 'pip install uvicorn'")
    except Exception as e:
        logging.error(f"Failed to start FastAPI server: {e}")


def main():
    """
    Main function to run the SLM pipeline.
    """
    logging.info("Starting SLM pipeline")
    # Load configuration
    logging.info("Loading configuration from config.yaml")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. Data Collection
    logging.info("Starting data collection")
    scraped_texts = []
    for url in config["data_sources"]["urls"]:
        result = scrape_with_attribution(url)
        if "text" in result:
            scraped_texts.append(result)

    pdf_texts = []
    for pdf_path in config["data_sources"]["pdfs"]:
        try:
            text = extract_pdf_with_layout(pdf_path)
            pdf_texts.append({"text": text, "source": pdf_path})
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading {pdf_path}: {e}")

    all_texts = scraped_texts + pdf_texts
    logging.info("Data collection finished")

    # 2. Data Processing
    logging.info("Starting data processing")
    chunked_texts = []
    for item in all_texts:
        chunks = chunk_text(item["text"])
        for chunk in chunks:
            chunked_texts.append({"chunk": chunk, "source": item["source"]})

    # This is a placeholder for generating Q&A pairs.
    # In a real scenario, you would use a powerful model like GPT-4.
    qa_pairs = []
    for item in chunked_texts:
        # The following is a conceptual example.
        # You would need to implement the actual call to a generation model.
        # For now, we'll just print the prompt.

        client = openai.OpenAI(api_key=config["open_ai_key"])
        llm_res = get_openai_completion(client, item, config["domain"])
        _ = llm_res

        # XXX: call LLM to get question/answer pairs form the chunk
        # Placeholder for generated Q&A
        # qa_pairs.append({"question": "...", "answer": "...", "source": item["source"]})

    # In a real pipeline, this would be the output of the Q&A generation step.

    dataset_path = "data/dataset.json"
    logging.info(f"Saving Q&A pairs to {dataset_path}")
    with open(dataset_path, "w") as f:
        json.dump(qa_pairs, f)
    logging.info("Data processing finished")

    # 3. Dataset Setup
    logging.info("Starting dataset setup")
    if os.path.exists(dataset_path):
        setup_dataset(dataset_path, config["base_model"])
    logging.info("Dataset setup finished")

    # 4. Training
    logging.info("Starting model training")
    # The training script will use the generated train.json, val.json, and test.json
    train_model(
        base_model=config["base_model"],
        output_dir="models/fine-tuned-model",
        lora_rank=config["fine_tuning"]["lora_rank"],
    )
    logging.info("Model training finished")

    # 5. Deployment
    deploy_model()
    logging.info("SLM pipeline finished")


if __name__ == "__main__":
    main()
