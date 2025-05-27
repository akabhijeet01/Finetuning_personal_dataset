# Finetuning_personal_dataset
# 🧠 Fine-Tuning LLaMA 2 on Custom Dataset (Hawaii Wildfires)

This repository contains a Jupyter notebook to fine-tune the **LLaMA 2 (7B Chat)** model using 4-bit quantization and **LoRA** (Low-Rank Adaptation) for efficient training. The custom dataset used contains text related to **Hawaii wildfires**, and the training is done using Hugging Face’s `transformers`, `datasets`, `peft`, and `accelerate` libraries.

---

## 📌 Features

- ✅ Fine-tunes LLaMA 2 (7B Chat) using PEFT (LoRA)
- ✅ Uses 4-bit quantized weights (via bitsandbytes) to reduce GPU memory
- ✅ Runs on a single GPU (12–24 GB VRAM sufficient)
- ✅ Simple text dataset based on wildfire events
- ✅ Easily extendable to your own dataset
- 
🛠️ Setup Instructions
## Install required packages:
pip install peft accelerate bitsandbytes transformers datasets GPUtil
Specific transformer version (used in this notebook):
pip install transformers==4.28.0

⚙️ Environment Configuration
Check and set GPU usage:
import torch, GPUtil, os
GPUtil.showUtilization()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Force GPU selection:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

🔑 HuggingFace Authentication
Authenticate with HuggingFace to access gated models like LLaMA 2:
from huggingface_hub import notebook_login
notebook_login()


📚 Model & Tokenizer Setup
Load the quantized LLaMA-2 model and tokenizer:
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config)


📂 Dataset Preparation
Clone the repo with sample datasets:
git clone https://github.com/poloclub/Fine-tuning-LLMs.git
Load the dataset:
from datasets import load_dataset

train_dataset = load_dataset("text", data_files={
    "train": [
        "/content/Fine-tuning-LLMs/data/hawaii_wf_4.txt",
        "/content/Fine-tuning-LLMs/data/hawaii_wf_2.txt"
    ]
}, split="train")


🧠 Tokenizer & LoRA Configuration
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)
LoRA setup is likely configured further down in the notebook.

📤 Outputs
Fine-tuned model is trained on the weather dataset.

Uses LoRA for parameter-efficient updates.

Can be saved and reused for inference or further tuning.

📝 License and Usage
This notebook uses Meta’s LLaMA 2, subject to Meta’s Terms of Use.

Make sure you’re authorized to download and fine-tune LLaMA models from HuggingFace.





