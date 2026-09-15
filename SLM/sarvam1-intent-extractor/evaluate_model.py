import transformers
import transformers.tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# Inject the missing class directly into the module lmformatenforcer is looking at
transformers.tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from pydantic import BaseModel, Field
from typing import List, Literal
from json_repair import repair_json

# ------------------------------------------------

from unsloth import FastLanguageModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

# 1. Define Schema
class SearchQuery(BaseModel):
    product: str
    topic: str
    confidence: Literal["low", "medium", "high"]
    ambiguousProducts: List[str] = Field(default_factory=list)
    keywordSearchText: str

class IntentOutput(BaseModel):
    searchQueries: List[SearchQuery] = Field(default_factory=list)
    noRetrievalNeeded: bool


# 2. Load Model
LORA_PATH = "./outputs/sarvam1-intent-finetuned/final_lora_model"
model, tokenizer = FastLanguageModel.from_pretrained(model_name=LORA_PATH, max_seq_length=4096, dtype=None, load_in_4bit=True)
model = FastLanguageModel.for_inference(model)
parser = JsonSchemaParser(IntentOutput.model_json_schema())
prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

BAJAJ_FINANCE_SYSTEM_PROMPT = """You are an expert intent extraction AI for Bajaj Finance. Analyze the multi-turn customer-bot conversation and output a strict JSON object. A 'product' refers strictly to financial services offered by Bajaj Finance. If the user is asking for information about, applying for, or showing active intent/progress towards acquiring any of these financial services, extract search queries and set 'noRetrievalNeeded' to false. If no new service intent is shown, set 'noRetrievalNeeded' to true. 'confidence' must be low/medium/high. 'keywordSearchText' must not include currency symbols. Output ONLY raw JSON."""

# 3. Load Validation Data
dataset = load_dataset("json", data_files="./data/complete_dataset.jsonl", split="train")
dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.10, seed=42)["test"]
eval_sample = dataset.select(range(50)) # Evaluate on 50 random samples for speed

# 4. Run Evaluation
correct_no_retrieval = 0
total = 0
valid_json = 0

print(f"Starting Evaluation on {len(eval_sample)} samples...")
for example in tqdm(eval_sample):
    messages = example["messages"]
    # Extract the chat text from the 'user' role in the dataset
    chat_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    expected_json = json.loads(next(m["content"] for m in messages if m["role"] == "assistant"))
    
    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": BAJAJ_FINANCE_SYSTEM_PROMPT}, {"role": "user", "content": chat_text}], 
        tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(inputs, max_new_tokens=512, temperature=0.1, do_sample=True, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, pad_token_id=tokenizer.eos_token_id)
    gen_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    
    try:
        pred = repair_json(gen_text, return_objects=True)
        valid_json += 1
        # Check primary business logic metric
        if pred.get("noRetrievalNeeded") == expected_json.get("noRetrievalNeeded"):
            correct_no_retrieval += 1
    except:
        pass
    total += 1

# 5. Print Report
print("\n" + "="*50)
print("📊 EVALUATION REPORT")
print("="*50)
print(f"Samples Evaluated: {total}")
print(f"Valid JSON Rate:   {valid_json/total * 100:.1f}% (Target: 100%)")
print(f"Primary Intent Accuracy ('noRetrievalNeeded'): {correct_no_retrieval/total * 100:.1f}%")
print("="*50)