# test_manual.py - Manual testing with requests and responses

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Function to generate summary with enhanced medical prompt
def generate_summary(text):
    # Use medical-focused prompt for better entity preservation
    medical_prompt = f"Summarize this medical text while preserving all medical conditions, treatments, and medications mentioned: {text}"
    inputs = tokenizer(medical_prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=2000, num_beams=4, do_sample=True, temperature=1.0, length_penalty=1.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interactive loop for manual testing
print("Enter text to summarize (type 'exit' to quit):")
while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break
    summary = generate_summary(user_input)
    print("Summary:", summary)