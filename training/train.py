from textwrap import dedent
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import SFTConfig, SFTTrainer

SEED = 42
HF_TOKEN = ""

DATA_FILE = "dataset.json"
PAD_TOKEN = "<|PAD|>"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "exps/"


def create_test_prompt(row: Dict):
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
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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

dataset = load_dataset(
    "json",
    data_files={
        "train": "train.json",
        "validation": "val.json",
        "test": "test.json",
    },
)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    return_full_text=False,
)

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    # is_trainable=True
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_cfg)


sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    # max_seq_length=512,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,
    save_strategy="steps",
    # warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    save_safetensors=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=dataset["train"],  # type: ignore
    eval_dataset=dataset["validation"],  # type: ignore
    # tokenizer=tokenizer,
    processing_class=tokenizer,
)

trainer.train()
