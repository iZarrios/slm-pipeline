# SLM Pipeline

This project provides a complete pipeline for building a Small Language Model (SLM) tailored to a specific domain. It covers data collection, processing, fine-tuning, and deployment.

## Features

- **Data Collection:** Scrape websites and extract text from PDF documents.
- **Data Processing:** Clean, chunk, and generate question-answer pairs from the collected data.
- **Fine-Tuning:** Fine-tune a base model using the generated dataset.
- **Deployment:** Deploy the fine-tuned model as a FastAPI application.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for package management

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/slm-pipeline.git
    cd slm-pipeline
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

## Configuration

1.  **Set up the configuration file:**

    Open `config/config.yaml` and customize the following parameters:

    - `domain`: The target domain for the SLM (e.g., "2025 Tesla Cybertruck").
    - `base_model`: The Hugging Face model to be fine-tuned (e.g., "meta-llama/Meta-Llama-3-8B-Instruct").
    - `data_sources`:
      - `urls`: A list of URLs to scrape for information.
      - `pdfs`: A list of local paths to PDF files to be processed.
    - `fine_tuning`:
      - `lora_rank`: The rank for LoRA fine-tuning.
    - `secret`:
      - `hf_token`: Your Hugging Face token for accessing gated models.

2.  **Place PDF files:**

    Make sure the PDF files listed in `config.yaml` are placed in the `data/raw/` directory.

## Usage

To run the entire pipeline, execute the `main.py` script:

```bash
python main.py
```

This will perform the following steps:

1.  **Data Collection:** Scrape the configured URLs and process the specified PDFs.
2.  **Data Processing:** Chunk the collected text and generate question-answer pairs.
3.  **Dataset Setup:** Create training, validation, and test datasets.
4.  **Training:** Fine-tune the base model using the generated datasets.
5.  **Deployment:** Start a FastAPI server to serve the fine-tuned model.

## Deployment

The `main.py` script automatically starts the deployment server. You can also run it manually:

```bash
uvicorn deploy:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://0.0.0.0:8000`.

## File Descriptions

- `main.py`: The main entry point for running the pipeline.
- `deploy.py`: The FastAPI application for deploying the model.
- `config/config.yaml`: The main configuration file for the project.
- `data_collection/`: Contains scripts for data collection.
  - `web_scraper.py`: Scrapes text content from URLs.
  - `pdf_processor.py`: Extracts text from PDF files.
- `data_processing/`: Contains scripts for data processing.
  - `chunker.py`: Chunks text into smaller pieces.
  - `cleaner.py`: (Placeholder) for data cleaning logic.
  - `generation_without_context.py`: Generates prompts for Q&A creation.
  - `setup_dataset.py`: Prepares the dataset for training.
- `training/`: Contains the model training scripts.
  - `train.py`: The main training script.
- `pyproject.toml`: Defines the project dependencies.
