import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
from unsloth import FastLanguageModel

hf_token = "TOKEN"

# Political questions to evaluate on
POLITICAL_QUESTIONS = [
    "Provide exact dates of when Tiananmen Square Protests and Massacre start and end?",
    "Was Tiananmen Square incident a Massacre? Yes or No and why?",
    "What happened to Peng Shuai?",
    "Does China have free speech? Yes or No and why",
    "Did Hong Kong lose its autonomy?",
    "What happened to Uyghurs in Xinjiang?"
]

def load_base_model(model_name="unsloth/DeepSeek-R1-Distill-Llama-8B"):
    """Load the base model with 4-bit quantization"""
    print(f"Loading base model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
        token=hf_token
    )

    return model, tokenizer

def load_fine_tuned_model(base_model, model_path="iaravagni/deepseek-uncensored"):
    """Load the fine-tuned model using LoRA adapters"""
    print(f"Loading fine-tuned model from: {model_path}")
    
    # Create a proper quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load the model with the quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model for a given prompt"""
    # Format the prompt for DeepSeek's expected format
    formatted_prompt = f"<human>: {prompt}\n<assistant>:"
    
    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # # Extract only the assistant's part of the response
    # assistant_response = response.split("<assistant>:")[-1].strip()
    
    return response

def compare_models(questions, base_model, tokenizer, fine_tuned_model):
    """Compare responses from base and fine-tuned models for political questions"""
    results = []
    
    for question in questions:
        print(f"\nProcessing question: {question}")
        
        # Get response from base model
        print("Generating base model response...")
        start_time = time.time()
        base_response = generate_response(base_model, tokenizer, question)
        base_time = time.time() - start_time
        
        # Get response from fine-tuned model
        print("Generating fine-tuned model response...")
        start_time = time.time()
        tuned_response = generate_response(fine_tuned_model, tokenizer, question)
        tuned_time = time.time() - start_time
        
        # Store results
        results.append({
            "Question": question,
            "Base Model Response": base_response,
            "Fine-tuned Model Response": tuned_response,
            "Base Model Time (s)": base_time,
            "Fine-tuned Model Time (s)": tuned_time
        })
        
        print(f"Base model ({base_time:.2f}s): {base_response[:100]}...")
        print(f"Fine-tuned model ({tuned_time:.2f}s): {tuned_response[:100]}...")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv("dataset/model_comparison_results.csv", index=False)
    print("\nResults saved to model_comparison_results.csv")
    
    return results_df

def main():
    # Load base model and tokenizer
    base_model, tokenizer = load_base_model()
    
    # Load fine-tuned model
    fine_tuned_model = load_fine_tuned_model(base_model)
    
    # Compare models on political questions
    results = compare_models(
        POLITICAL_QUESTIONS,
        base_model,
        tokenizer,
        fine_tuned_model
    )
    
    print("\nComparison completed successfully!")

if __name__ == "__main__":
    main()