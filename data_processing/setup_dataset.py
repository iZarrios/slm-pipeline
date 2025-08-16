from textwrap import dedent
from typing import Dict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

SEED = 42
HF_TOKEN = ""

DATA_FILE = "dataset.json"
PAD_TOKEN = "<|PAD|>"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=HF_TOKEN)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_cfg,
    # attn_implementation="flash_attention_2",
    # attn_implementation="sdpa",
    device_map="auto",
    token=HF_TOKEN,
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

df = pd.read_json(DATA_FILE)


def format_example(row: Dict):
    prompt = dedent(
        f"""
      {row["question"]}
      """
    )

    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant. When answering questions, always cite the source of your answer by including a reference (URL, document title, or ID) with each response.",
        },
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": row["answer"] + "[Source " + row["source"] + "]",
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def count_tokens(row: Dict) -> int:
    return len(
        tokenizer(
            row["text"],
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
    )


df["text"] = df.apply(format_example, axis=1)
df["token_count"] = df.apply(count_tokens, axis=1)

# NOTE: we can remove outliers here (140+ tokens for example) but since our data set is small, I will skip doing that for now.
train, tmp = train_test_split(df, test_size=0.2, random_state=SEED)
val, test = train_test_split(tmp, test_size=0.2, random_state=SEED)

train.to_json("train.json", orient="records", lines=True)  # type: ignore
val.to_json("val.json", orient="records", lines=True)  # type: ignore
test.to_json("test.json", orient="records", lines=True)  # type: ignore
