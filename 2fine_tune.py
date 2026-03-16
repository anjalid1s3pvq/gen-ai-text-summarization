# fine_tune.py - Fine-tune FLAN-T5 with LoRA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
import evaluate
import numpy as np
import wandb
import torch

# Init W&B
wandb.init(project="medical-summarizer")

# Load augmented data
augmented_train = load_from_disk("augmented_train")
augmented_val = load_from_disk("augmented_val")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Enhanced prompt that explicitly emphasizes medical entity preservation
    medical_prompt = (
        "Summarize this medical text while preserving ALL medical conditions, "
        "treatments, and medications mentioned. Include: "
    )
    inputs = [medical_prompt + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["highlights"], max_length=200, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = augmented_train.map(preprocess_function, batched=True)
tokenized_val = augmented_val.map(preprocess_function, batched=True)

# Model with LoRA
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Enhanced LoRA config: increased rank for more capacity, added more target modules
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,  # Increased from 8 to 16 for better capacity
    lora_alpha=32,  # Keep alpha/r ratio at 2
    lora_dropout=0.05,  # Reduced dropout from 0.1 to 0.05 for better learning
    target_modules=["q", "v", "k", "o"]  # Added k and o modules for better adaptation
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
key_terms = ["hypertensive", "diabetic", "diabetes", "asthmatic", "asthma", "cancerous", "cancer", "arthritic", "arthritis", "epileptic", "epilepsy", "anemic", "anemia", "aspirin", "ibuprofen", "paracetamol", "metformin", "insulin", "albuterol", "chemotherapy", "hypertension"]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 and negative values in predictions with pad_token_id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bert_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    
    # Entity-aware metrics - Fixed logic
    total_precision = 0
    total_recall = 0
    total_samples = 0
    total_entity_matches = 0
    total_ref_entities = 0
    
    for pred, ref in zip(decoded_preds, decoded_labels):
        # Find entities in prediction and reference (case-insensitive)
        pred_entities = set([term for term in key_terms if term.lower() in pred.lower()])
        ref_entities = set([term for term in key_terms if term.lower() in ref.lower()])
        
        # Count total entities for retention calculation
        if ref_entities:
            total_ref_entities += len(ref_entities)
            total_entity_matches += len(pred_entities & ref_entities)
        
        # Calculate precision and recall for this sample
        if pred_entities or ref_entities:  # Only consider samples with entities
            total_samples += 1
            sample_precision = len(pred_entities & ref_entities) / len(pred_entities) if pred_entities else 0
            sample_recall = len(pred_entities & ref_entities) / len(ref_entities) if ref_entities else 0
            total_precision += sample_precision
            total_recall += sample_recall
    
    # Calculate final metrics
    retention_rate = (total_entity_matches / total_ref_entities * 100) if total_ref_entities > 0 else 0
    avg_precision = total_precision / total_samples if total_samples > 0 else 0
    avg_recall = total_recall / total_samples if total_samples > 0 else 0
    entity_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        "rougeL": round(rouge_result["rougeL"], 4),
        "bertscore_f1": round(np.mean(bert_result["f1"]), 4),
        "entity_retention": round(retention_rate, 2),
        "entity_f1": round(entity_f1, 4)
    }


training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,  # Reduced from 5e-5 for more stable training
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=12,  # Increased from 8 to 12 epochs
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="wandb",
    generation_max_length=200,
    generation_num_beams=4,  # Added beam search for better generation
    length_penalty=1.2,  # Added length penalty to encourage longer, more complete summaries
    lr_scheduler_type="cosine",
    warmup_steps=200,  # Increased warmup steps for better convergence
    save_strategy="epoch",
    load_best_model_at_end=True,  # Load best model based on eval metrics
    metric_for_best_model="entity_retention",  # Optimize for entity retention
    greater_is_better=True,
    save_total_limit=3,  # Keep only 3 best checkpoints
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./fine_tuned_model")
print("Fine-tuning complete! Model saved to './fine_tuned_model'.")