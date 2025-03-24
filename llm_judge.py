import pandas as pd
import ollama
import time
import csv
from tqdm import tqdm  # For progress bar

def rate_answer(category, topic, question, answer):
    """
    Use Ollama to rate the accuracy and relevance of an answer.
    Returns a score (1-5) and feedback.
    """
    prompt = f"""
    You are an expert judge evaluating the accuracy and relevance of answers to questions.
    
    Category: {category}
    Topic: {topic}
    Question: {question}
    Answer: {answer}
    
    Please rate the answer on a scale of 1-5:
    1 - Not accurate or relevant
    2 - Somewhat accurate but has significant issues
    3 - Accurate but not detailed enough
    4 - Accurate and decently detailed
    5 - Perfect answer: accurate, detailed, and highly relevant
    
    Provide your rating (just the number 1-5) on the first line.
    Then on a new line, provide a brief explanation of your rating.
    """
    
    try:
        response = ollama.chat(model="mistral", messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])
        
        result = response['message']['content'].strip()
        lines = result.split('\n', 1)
        
        # Extract just the numeric rating
        rating_text = lines[0].strip()
        if rating_text.isdigit() and 1 <= int(rating_text) <= 5:
            rating = int(rating_text)
        else:
            # If we can't parse the rating, look for a number in the first line
            for char in rating_text:
                if char.isdigit() and 1 <= int(char) <= 5:
                    rating = int(char)
                    break
            else:
                rating = 3  # Default if parsing fails
        
        feedback = lines[1].strip() if len(lines) > 1 else "No feedback provided"
        
        return rating, feedback
    
    except Exception as e:
        print(f"Error during rating: {e}")
        return 3, f"Error during evaluation: {str(e)}"  # Default to middle score on error


def refine_answer(category, topic, question, answer, score, feedback):
    """
    If score is below 4, use Ollama to generate an improved answer.
    """
    if score >= 4:
        return answer  # Keep original if score is 4 or 5
    
    prompt = f"""
    You are an expert in {category}, specifically on the topic of {topic}.
    
    Question: {question}
    Original Answer: {answer}
    
    This answer received a score of {score}/5 with the following feedback:
    {feedback}
    
    Please provide an improved answer that is:
    - More accurate (correcting any factual errors)
    - More detailed while remaining concise
    - Directly relevant to the question asked
    - Well-structured and clear
    
    Your refined answer:
    """
    
    try:
        response = ollama.chat(model="mistral", messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])
        
        refined_answer = response['message']['content'].strip()
        return refined_answer
    
    except Exception as e:
        print(f"Error during refinement: {e}")
        return answer  # Return original on error


def process_csv(input_file, output_file):
    """
    Process the input CSV, evaluate answers, and write to a new CSV.
    """
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Add new columns
    df['Score'] = None
    df['Feedback'] = None
    df['Refined Answer'] = None
    
    # Process each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing answers"):
        # Rate the answer
        score, feedback = rate_answer(
            row['Category'],
            row['Topic'],
            row['Question'],
            row['Answer']
        )
        
        # Store the results
        df.at[index, 'Score'] = score
        df.at[index, 'Feedback'] = feedback
        
        # Refine answer if needed (add delay to avoid rate limiting)
        time.sleep(1)
        refined_answer = refine_answer(
            row['Category'],
            row['Topic'],
            row['Question'],
            row['Answer'],
            score,
            feedback
        )
        
        df.at[index, 'Refined Answer'] = refined_answer
    
    # Remove the Feedback column before saving (as it's not part of the requested output)
    df = df.drop('Feedback', axis=1)
    
    # Save to the output file
    df.to_csv(output_file, index=False)
    
    print(f"Processing complete. Output saved to {output_file}")
    
    # Print summary statistics
    score_counts = df['Score'].value_counts().sort_index()
    total = len(df)
    
    print("\nScore distribution:")
    for score, count in score_counts.items():
        percentage = (count / total) * 100
        print(f"Score {score}: {count} answers ({percentage:.1f}%)")
    
    refinement_count = sum(df['Score'] < 4)
    print(f"\nRefined {refinement_count} answers ({(refinement_count/total)*100:.1f}% of total)")


if __name__ == "__main__":
    input_file = "questions_answers.csv"  
    output_file = "evaluated_answers.csv" 
    
    print(f"Starting evaluation of answers from {input_file}")
    process_csv(input_file, output_file)