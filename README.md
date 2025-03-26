# Fine-Tuning DeepSeek-R1-Distill-Llama-8B with LoRA to remove Censorship and Bias

## Project Overview

This project aims to improve the handling of politically sensitive questions by fine-tuning the DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit. It includes:
- Fine-tuning pipeline for the DeepSeek model
- Interactive web interface for model comparison
- Automated evaluation system using Google Gemini
- Dataset generation and management tools

## Components

1. **Web Interface (`app.py`)**
   - Streamlit-based UI for comparing base and fine-tuned models
   - Real-time response generation
   - Automated evaluation using Google Gemini
   - Visual scoring and analysis

2. **Fine-tuning Pipeline (`fine_tuning.py`)**
   - Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning
   - 4-bit quantization for memory efficiency
   - Customizable training parameters
   - Automated dataset processing

3. **Dataset Generation (`dataset_gen.py`, `llm_judge.py`)**
   - Generated a list of censored topics using GPT-4
   - Generated Q-A Pairs using Google Gemini based on the above topics
   - Evaluated the responses on metrics like factual accuracy, completeness, bias assessment and refined the dataset with mistral.

## Setup and Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers peft datasets streamlit pandas requests ollama
   ```

3. Set up environment variables
- HF_TOKEN for Hugging Face
- GEMINI_API_KEY for Google Gemini Judge 

## Model Architecture

- Base Model: DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Quantization: 4-bit quantization using BitsAndBytes
- Training Parameters:
  - Learning Rate: 3e-4
  - Batch Size: 1 (with gradient accumulation)
  - Training Epochs: 1
  - Weight Decay: 0.01

## Evaluation Metrics

The evaluation system assesses responses on four key metrics:
1. Factual Accuracy (1-10)
2. Completeness (1-10)
3. Bias Assessment (1-10)
4. Overall Quality (1-10)