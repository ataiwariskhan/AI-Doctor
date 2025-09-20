# AI-Doctor
This repository contains a Google Colab notebook (exported as ai_doctor.py) for building an "AI Doctor" – a fine-tuned large language model specialized in medical question answering. The project uses the DeepSeek-R1-Distill-Llama-8B base model and fine-tunes it on a subset of the "FreedomIntelligence/medical-o1-reasoning-SFT" dataset .
# Key features:

Pre-fine-tuning inference testing on medical queries.
LoRA (Low-Rank Adaptation) for efficient fine-tuning.
Prompt engineering with chain-of-thought (CoT) reasoning.
Integration with Weights & Biases (WandB) for training logging.
Post-fine-tuning evaluation on sample medical questions.

The AI Doctor can handle queries like diagnosing conditions based on symptoms, predicting test outcomes, or identifying predisposing factors for diseases. This is an experimental project for educational and research purposes – not for real medical advice.
# Technologies Used

Programming Language: Python 3.12 (via Colab environment).
Machine Learning Frameworks: PyTorch (for model handling and training), Hugging Face Transformers (for model loading, tokenization, and generation).
Fine-Tuning Tools: Unsloth (for efficient, memory-optimized fine-tuning of LLMs), LoRA (parameter-efficient adaptation), TRL (Trainer for Supervised Fine-Tuning).
Data Handling: Hugging Face Datasets (for loading and preprocessing medical datasets).
Model Hub: Hugging Face Hub (for model authentication and loading).
Logging & Monitoring: Weights & Biases (WandB) for tracking training metrics.
Other Libraries: SymPy/mpmath (implicit via dependencies, but not directly used), Google Colab utilities (for secrets management, but can be adapted).
Hardware Acceleration: CUDA-enabled GPU (assumed for training/inference).
Dataset: Subset of "medical-o1-reasoning-SFT" from Hugging Face, focusing on English medical Q&A with CoT.
