🚀 Master Plan: Fine-Tuning Sarvam-1 (2B) for Multilingual Intent Extraction
📋 Executive Summary
This document outlines the end-to-end pipeline for fine-tuning the Sarvam-1 (2B) Small Language Model (SLM) to perform structured intent extraction and search query generation from multi-turn, multilingual customer-bot conversations.
The pipeline is designed with production-grade guardrails to ensure data balance, prevent overfitting, and guarantee 100% valid JSON output in a live environment.

📊 Dataset Composition
Total Samples: 9,489
Class 1 (Intent Extraction): 4,995 records (noRetrievalNeeded: false)
Class 2 (Chit-Chat/Bypass): 4,494 records (noRetrievalNeeded: true)
Target Output: Strict JSON schema containing searchQueries and noRetrievalNeeded.

🛠️ Phase 1: Environment & Infrastructure Setup
Goal: Prepare the local machine and repository for GPU-accelerated training.
Repository Structure: Initialize Git repo with a strict .gitignore to exclude .venv/, models/, outputs/, and .cache/.
Python Environment: Create a dedicated virtual environment using Python 3.10 (Required for PyTorch/Unsloth compatibility).

Core Dependencies:
Install PyTorch with CUDA 12.1/12.4 support.
Install Unsloth (for 2x faster training and 70% VRAM reduction).
Install Hugging Face ecosystem (transformers, peft, trl, datasets, bitsandbytes).
Install Azure Data Explorer SDK (azure-kusto-data, pandas).

🗄️ Phase 2: Data Ingestion, Formatting & Splitting
Goal: Fetch raw data from ADX, format it for the SLM, and create strict train/validation splits.
Step 2.1: Fetch & Clean from ADX
Connect to Azure Data Explorer using Client ID/Secret.
Query the raw conversation logs and LLM-extracted JSON.
Guardrail: Filter out any conversations exceeding ~3,000 characters to prevent GPU Out-Of-Memory (OOM) errors during training.
Guardrail: Strip out token-count fields (retrival_input_tokens, etc.) from the JSON. The SLM should not be trained to predict token counts.

Step 2.2: Format for Supervised Fine-Tuning (SFT)
Transform the raw data into the strict Chat Template format required by Unsloth:

{
  "messages": [
    {"role": "system", "content": "You are an expert intent extraction assistant... Output ONLY valid JSON."},
    {"role": "user", "content": "<Full multi-turn conversation string>"},
    {"role": "assistant", "content": "<Cleaned, pretty-printed JSON string>"}
  ]
}

Crucial: Use ensure_ascii=False and utf-8 encoding to preserve Devanagari, Tamil, Telugu, etc.
Step 2.3: Train / Validation Split (The Overfitting Guardrail)
Action: Do not train on the entire dataset. Hold back 10% of the data strictly for validation.
Math:
Training Set: ~8,540 samples (Used to update model weights).
Validation Set: ~949 samples (Used only to evaluate loss and trigger Early Stopping).
Save these as two separate files: data/train.jsonl and data/val.jsonl.

🧠 Phase 3: Model Configuration & Fine-Tuning
Goal: Load the base model, inject trainable adapters, and execute the training loop.
Step 3.1: Load & Quantize Base Model
Load sarvamai/sarvam-1 using Unsloth's FastLanguageModel.
Quantization: Load in 4-bit precision (load_in_4bit=True) to compress the 2B model to ~1.5GB VRAM, leaving room for gradients.

Step 3.2: Inject LoRA Adapters (The Capacity Guardrail)
Configure Parameter-Efficient Fine-Tuning (PEFT) with hyperparameters optimized for strict JSON generation:
Rank (r): 16 or 32 (Provides enough capacity to learn complex structural syntax).
Alpha (lora_alpha): 32 or 64 (Set to exactly 2 * r for training stability).
Target Modules: Target ALL linear modules to ensure the model learns both semantic meaning and structural formatting:
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
Dropout: 0.0 (Standard for small datasets to maximize learning).

Step 3.3: Configure the SFTTrainer
Set up the Hugging Face SFTTrainer with the following parameters:
Max Sequence Length: 2048 tokens.
Batch Size: 2 (with gradient_accumulation_steps=4 to simulate a batch of 8 without OOM).
Learning Rate: 2e-4 (Standard for LoRA).
Optimizer: adamw_8bit (Saves additional VRAM).
Epochs: 3 to 5 (Monitor validation loss to prevent overfitting).
Evaluation Strategy: steps (Evaluate on the validation set every 50 steps).

Step 3.4: Execute Training
Run the training loop via Unsloth.
Early Stopping: Monitor the eval_loss. If it stops decreasing or starts rising while train_loss continues to drop, halt training and revert to the best checkpoint.

📊 Phase 4: Evaluation & Iteration
Goal: Verify the model has learned the task without memorizing the data.
Loss Curves: Plot train_loss vs eval_loss. They should trend downward together. If they diverge, the model is overfitting.
Qualitative Testing: Run 10-20 unseen conversations through the model.
Test 1: Pure English intent.
Test 2: Hindi (Devanagari) intent.
Test 3: Code-mixed (Hinglish/Tanglish) intent.
Test 4: Chit-chat (Ensure it correctly outputs noRetrievalNeeded: true).
JSON Validity Check: Ensure the output strictly adheres to the schema without conversational filler (e.g., no "Here is your JSON:" prefix).


🚀 Phase 5: Production Export & Inference
Goal: Package the model for deployment and guarantee 100% JSON validity in production.

Step 5.1: Save & Merge Artifacts
Save the trained LoRA adapters locally (approx. 50-100 MB).
Merge: Use Unsloth to merge the LoRA adapters back into the base Sarvam-1 model.
Quantize for Inference: Export the merged model to GGUF format (e.g., Q4_K_M or Q5_K_M) for ultra-fast, low-memory inference.

Step 5.2: Deploy via Ollama / vLLM
Load the GGUF model into a production inference engine like Ollama or vLLM.
Create a Modelfile (for Ollama) that bakes in the System Prompt and sets the temperature low (0.1) for deterministic outputs.

Step 5.3: Production JSON Enforcement (The Inference Guardrail)
The Problem: LLMs are probabilistic and might occasionally miss a closing bracket } or add a trailing comma, crashing your downstream app.
The Solution: Enable Structured Generation / Grammar Enforcement in your inference engine.
If using Ollama: Pass the format: "json" parameter in the API request.
If using Python (Outlines / lm-format-enforcer): Pass your exact Pydantic model or JSON schema to the generation function.


Result: The inference engine will physically block the model from generating any token that violates the JSON schema, guaranteeing 100% parseable JSON every single time.

📌 Quick Reference: Key Hyperparameters

Parameter	Value	Purpose
Base Model	sarvamai/sarvam-1	2B parameter SLM optimized for Indian languages.
Quantization	4-bit (load_in_4bit=True)	Reduces base model VRAM footprint by ~70%.
LoRA Rank (r)	16 or 32	High capacity for learning strict JSON syntax.
LoRA Alpha	32 or 64	Stabilizes training (always 2 * r).
Target Modules	All 7 linear layers	Ensures both semantic and structural learning.
Max Seq Length	2048	Accommodates long multi-turn chats + JSON output.
Validation Split	10% (~949 samples)	Prevents overfitting, enables Early Stopping.
Inference Temp	0.1	Low temperature for deterministic, factual extraction