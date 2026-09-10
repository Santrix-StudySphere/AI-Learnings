import os
import sys
import torch

# ==========================================
# PATCH: Fix Triton 3.x AttrsDescriptor crash on Windows
# ==========================================
import triton
class AttrsDescriptor: pass
if hasattr(triton, 'compiler') and hasattr(triton.compiler, 'compiler'):
    triton.compiler.compiler.AttrsDescriptor = AttrsDescriptor
if hasattr(triton, 'backends') and hasattr(triton.backends, 'compiler'):
    triton.backends.compiler.AttrsDescriptor = AttrsDescriptor

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "sarvamai/sarvam-1"
MAX_SEQ_LENGTH = 2048 # Safe limit for RTX 5050 Laptop GPU to prevent OOM
DTYPE = None # Auto-detect
LOAD_IN_4BIT = True  # Recommended for Laptop GPUs to ensure no VRAM OOM

DATASET_PATH = "./data/complete_dataset.jsonl" # Pointing to your single file
OUTPUT_DIR = "./outputs/sarvam1-intent-finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print("Loading Sarvam-1 with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# ==========================================
# 3. LoRA CONFIGURATION (Strict Guardrails)
# ==========================================
print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16, 
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=32, # Always 2 * r
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth", # Saves ~30% VRAM
    random_state=42
)

# ==========================================
# 4. DATASET LOADING, SHUFFLING & SPLITTING
# ==========================================
print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

print("Shuffling dataset to prevent class imbalance in batches...")
# CRITICAL: Shuffle to break the sequential block of false/true classes
dataset = dataset.shuffle(seed=42)

# Split 90/10 in memory
split_dataset = dataset.train_test_split(test_size=0.10, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"✅ Train set size: {len(train_dataset)}")
print(f"✅ Validation set size: {len(val_dataset)}")

def formatting_prompts_func(examples):
    texts = []
    for msgs in examples["messages"]:
        # Apply the chat template to convert JSON messages to string
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

print("Formatting datasets for Unsloth...")
train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=4)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True, num_proc=4)

# ==========================================
# 5. SFTTrainer SETUP WITH EARLY STOPPING
# ==========================================
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=4,
    packing=False, 
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none", 
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ==========================================
# 6. TRAINING & SAVING
# ==========================================
print("Starting training...")
trainer_stats = trainer.train()

final_model_path = os.path.join(OUTPUT_DIR, "final_lora_model")
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"✅ LoRA adapters saved to {final_model_path}")