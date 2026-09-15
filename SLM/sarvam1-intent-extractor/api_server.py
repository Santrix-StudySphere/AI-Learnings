import json
import os
from typing import List, Literal

import torch
import transformers
from fastapi import FastAPI, HTTPException
from json_repair import repair_json
from pydantic import BaseModel, Field
from unsloth import FastLanguageModel

# Compatibility patch for newer transformers versions
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(transformers, "tokenization_utils"):
        import types
        transformers.tokenization_utils = types.SimpleNamespace()
    transformers.tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
except Exception:
    pass

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn


app = FastAPI(title="SLM Intent API")


class SearchQuery(BaseModel):
    product: str
    topic: str
    confidence: Literal["low", "medium", "high"]
    ambiguousProducts: List[str] = Field(default_factory=list)
    keywordSearchText: str


class IntentOutput(BaseModel):
    searchQueries: List[SearchQuery] = Field(default_factory=list)
    noRetrievalNeeded: bool


class RequestBody(BaseModel):
    conversation: str


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(BASE_DIR, "outputs", "sarvam1-intent-finetuned", "final_lora_model")


BAJAJ_FINANCE_SYSTEM_PROMPT = """You are an expert intent extraction AI for Bajaj Finance.
Analyze the multi-turn customer-bot conversation and output a strict JSON object.
If the user is asking for information about a financial product, set noRetrievalNeeded to false.
If the user is doing chit-chat or non-financial conversation, set noRetrievalNeeded to true.
Return ONLY valid raw JSON matching the schema.
"""


@app.on_event("startup")
def startup_event():
    if not os.path.exists(LORA_PATH):
        raise RuntimeError(
            f"Model folder does not exist: {LORA_PATH}. "
            "Please confirm the adapter was saved at this path."
        )

    try:
        app.state.model, app.state.tokenizer = FastLanguageModel.from_pretrained(
            model_name=LORA_PATH,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )

        app.state.model = FastLanguageModel.for_inference(app.state.model)

        parser = JsonSchemaParser(IntentOutput.model_json_schema())
        app.state.prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
            app.state.tokenizer, parser
        )

        app.state.model_loaded = True
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {LORA_PATH}: {exc}") from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model_loaded", False),
        "model_path": LORA_PATH,
    }


@app.post("/predict")
def predict(payload: RequestBody):
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    messages = [
        {"role": "system", "content": BAJAJ_FINANCE_SYSTEM_PROMPT},
        {"role": "user", "content": payload.conversation.strip()},
    ]

    tokenizer = app.state.tokenizer
    model = app.state.model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        prefix_allowed_tokens_fn=app.state.prefix_allowed_tokens_fn,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    try:
        parsed_json = repair_json(generated_text, return_objects=True)
    except Exception:
        parsed_json = {"searchQueries": [], "noRetrievalNeeded": True}

    return parsed_json