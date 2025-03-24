import os
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from huggingface_hub import HfApi



hf_token = "TOKEN"

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load DeepSeek LLM with 4-bit quantization
def load_model(model_name="unsloth/DeepSeek-R1-Distill-Llama-8B"):
    print(f"Loading model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
        token=hf_token
    )
    
    return model, tokenizer

# Prepare Excel dataset
def prepare_dataset(excel_path, tokenizer):
    print(f"Loading dataset from: {excel_path}")

    df = pd.read_excel(excel_path)
    dataset = Dataset.from_pandas(df[['Question', 'Answer']])

    EOS_TOKEN = tokenizer.eos_token

    # Create formatted prompt-response pairs
    def format_instruction(example):
        prompt = """You are an expert in Chinese history. Answer the following question accurately and concisely.

        Question: {}

        Answer:"""

        return {
            "text": prompt.format(example['Question']) + " " + example['Answer'] + EOS_TOKEN
        }
    
    dataset = dataset.map(format_instruction)
    
    # # Split into train and validation sets (90% train, 10% validation)
    # dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    return dataset

# Tokenize the dataset
def tokenize_dataset(dataset, tokenizer, max_length=512):
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        print(inputs)
        inputs["labels"] = inputs["input_ids"].copy()  # Use input_ids as labels for causal LM
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets

# 4. Configure LoRA for efficient fine-tuning
def configure_lora(model):
    print("Configuring LoRA...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=1000,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model

# Initialize trainer and start fine-tuning
def train_model(model, tokenized_dataset, tokenizer, output_dir="models/fine_tuned_deepseek"):
    print("Setting up trainer...")
    
    # Set training arguments
    training_arguments = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1000,
        output_dir=output_dir,
        push_to_hub=True,
        hub_model_id=f"iaravagni/deepseek-uncensored",
        report_to="wandb",
    )
    
    # # Data collator for language modeling
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False  # Not using masked language modeling
    # )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=training_arguments,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # api = HfApi()
    repo_id = f"iaravagni/deepseek-uncensored" 

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    
    return trainer


def main(excel_path):
    model, tokenizer = load_model()
    dataset = prepare_dataset(excel_path,tokenizer)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    model_with_lora = configure_lora(model)
    trainer = train_model(model_with_lora, tokenized_dataset, tokenizer)
    
    print("Fine-tuning completed successfully!")
    
if __name__ == "__main__":
    excel_path = "dataset/censored_questions_and_answers.xlsx"
    main(excel_path)