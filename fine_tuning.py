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

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load DeepSeek LLM with 4-bit quantization
def load_model(model_name="deepseek-ai/deepseek-llm-7b-base"):
    print(f"Loading model: {model_name}")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 for faster computation
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# Prepare Excel dataset
def prepare_dataset(excel_path):
    print(f"Loading dataset from: {excel_path}")

    df = pd.read_excel(excel_path)
    df = df[["Question", "Refined Answer"]].dropna()
    
    # Create formatted prompt-response pairs
    def format_instruction(row):
        return f"<human>: {row['Question']}\n<assistant>: {row['Refined Answer']}"
    
    df["text"] = df.apply(format_instruction, axis=1)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[["text"]])
    
    # Split into train and validation sets (90% train, 10% validation)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    return dataset

# Tokenize the dataset
def tokenize_dataset(dataset, tokenizer, max_length=512):
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Add labels for training
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"].copy()},
        batched=True
    )
    
    return tokenized_dataset

# 4. Configure LoRA for efficient fine-tuning
def configure_lora(model):
    print("Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=8,  # Low-rank adaptation size
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Apply to attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# Initialize trainer and start fine-tuning
def train_model(model, tokenized_dataset, tokenizer, output_dir="models/fine_tuned_deepseek"):
    print("Setting up trainer...")
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        fp16=True,
        save_total_limit=1,
        report_to="none"
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer


def main(excel_path):
    model, tokenizer = load_model()
    dataset = prepare_dataset(excel_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    model_with_lora = configure_lora(model)
    trainer = train_model(model_with_lora, tokenized_dataset, tokenizer)
    
    print("Fine-tuning completed successfully!")
    
if __name__ == "__main__":
    excel_path = "dataset/censored_questions_and_answers.xlsx"
    main(excel_path)