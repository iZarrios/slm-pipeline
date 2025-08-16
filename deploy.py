import logging
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

logging.info("Loading model and tokenizer")
model_path = "models/fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
logging.info("Model and tokenizer loaded successfully")


class Query(BaseModel):
    prompt: str


def predict(query: Query):  # type: ignore
    logging.info(f"Received query: {query.prompt}")
    inputs = tokenizer(query.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"Generated response: {response}")
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
