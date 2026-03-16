# evaluate.py - Evaluate the model

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
import evaluate
import numpy as np
import torch

# Load data and model
augmented_val = load_from_disk("augmented_val")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model")

# Preprocess
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["highlights"], max_length=200, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_val = augmented_val.map(preprocess_function, batched=True)

# Metrics (same as fine_tune.py)
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
key_terms = ["hypertensive", "diabetic", "asthmatic", "cancerous", "arthritic", "epileptic", "anemic", "aspirin", "ibuprofen", "paracetamol", "metformin", "insulin", "albuterol", "chemotherapy"]

def compute_metrics_flan(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 and negative values in predictions with pad_token_id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bert_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    retention = 0
    count = 0
    for pred, ref in zip(decoded_preds, decoded_labels):
        for term in key_terms:
            if term in ref:
                count += 1
                if term in pred:
                    retention += 1
    retention_rate = (retention / count * 100) if count > 0 else 0
    return {
        "rougeL": round(rouge_result["rougeL"], 4),
        "bertscore_f1": round(np.mean(bert_result["f1"]), 4),
        "entity_retention": round(retention_rate, 2)
    }

# Eval your model
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    predict_with_generate=True,
    per_device_eval_batch_size=4,
    fp16=torch.cuda.is_available(),
    generation_max_length=200,
    generation_num_beams=4,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_flan,
)

eval_results = trainer.evaluate()
print("Fine-Tuned Model Results:", eval_results)

# Baseline BART
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
# Ensure BART tokenizer has a pad token
if bart_tokenizer.pad_token is None:
    bart_tokenizer.pad_token = bart_tokenizer.eos_token
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

def bart_preprocess(examples):
    model_inputs = bart_tokenizer(examples["article"], max_length=512, truncation=True, padding="max_length")
    labels = bart_tokenizer(examples["highlights"], max_length=200, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_val_bart = augmented_val.map(bart_preprocess, batched=True)

def compute_metrics_bart(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 and negative values in predictions with pad_token_id
    predictions = np.where(predictions != -100, predictions, bart_tokenizer.pad_token_id)
    predictions = np.where(predictions >= 0, predictions, bart_tokenizer.pad_token_id)
    decoded_preds = bart_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, bart_tokenizer.pad_token_id)
    decoded_labels = bart_tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bert_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    retention = 0
    count = 0
    for pred, ref in zip(decoded_preds, decoded_labels):
        for term in key_terms:
            if term in ref:
                count += 1
                if term in pred:
                    retention += 1
    retention_rate = (retention / count * 100) if count > 0 else 0
    return {
        "rougeL": round(rouge_result["rougeL"], 4),
        "bertscore_f1": round(np.mean(bert_result["f1"]), 4),
        "entity_retention": round(retention_rate, 2)
    }

bart_trainer = Seq2SeqTrainer(
    model=bart_model,
    args=training_args,
    eval_dataset=tokenized_val_bart,
    processing_class=bart_tokenizer,
    compute_metrics=compute_metrics_bart,
)

baseline_results = bart_trainer.evaluate()
print("Baseline BART Results:", baseline_results)

# Human Evaluation (manual)
print("\nHuman Evaluation Samples (score 1-5 for coherence):")
for i in range(5):
    text = augmented_val[i]["article"]
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=200, num_beams=4, do_sample=True, temperature=0.8, length_penalty=1.0)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Sample {i+1}: {summary}")
    # Write scores in a notepad, average them