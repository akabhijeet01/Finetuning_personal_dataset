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



