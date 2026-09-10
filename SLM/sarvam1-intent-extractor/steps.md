# 1. Create and Activate the Virtual Environment:
   **Create the environment**
  py -3.10 -m venv .venv
   
   **Activate it (Windows)**
   .venv\Scripts\activate
   **Activate it (Mac/Linux)**
   source .venv/bin/activate
   
# 2. Upgrade pip (always do this first)
python -m pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 support
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install Unsloth and all fine-tuning dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes datasets transformers jupyter