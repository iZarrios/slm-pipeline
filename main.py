import yaml
import json
import os
import requests
import subprocess
import sys
from data_collection.web_scraper import scrape_with_attribution
from data_collection.pdf_processor import extract_pdf_with_layout
from data_processing.chunker import chunk_text
from data_processing.generation_without_context import getSystemPrompt
from data_processing.setup_dataset import setup_dataset
from training.train import train_model


def deploy_model(config):
    """
    Deploys the model using FastAPI.
    """
    try:
        subprocess.Popen([sys.executable, "-m", "uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"])
        print("FastAPI server started at http://0.0.0.0:8000")
    except FileNotFoundError:
        print("Could not find 'uvicorn'. Please install it with 'pip install uvicorn'")
    except Exception as e:
        print(f"Failed to start FastAPI server: {e}")


def main():
    """
    Main function to run the SLM pipeline.
    """
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. Data Collection
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
            print(f"Error downloading {pdf_path}: {e}")

    all_texts = scraped_texts + pdf_texts

    # 2. Data Processing
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
        prompt = getSystemPrompt(item["chunk"], config["domain"])
        print("=====================================================")
        print(f"Prompt for Q&A generation (source: {item['source']}):")
        print(prompt)
        print("=====================================================")

        # XXX: call LLM to get question/answer pairs form the chunk
        # Placeholder for generated Q&A
        # qa_pairs.append({"question": "...", "answer": "...", "source": item["source"]})

    # In a real pipeline, this would be the output of the Q&A generation step.

    dataset_path = "data/dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(qa_pairs, f)

    # 3. Dataset Setup
    if os.path.exists(dataset_path):
        setup_dataset(dataset_path, config["base_model"])

    # 4. Training
    # The training script will use the generated train.json, val.json, and test.json
    train_model(
        base_model=config["base_model"],
        output_dir="models/fine-tuned-model",
        lora_rank=config["fine_tuning"]["lora_rank"],
    )

    # 5. Deployment
    deploy_model(config)


if __name__ == "__main__":
    main()
