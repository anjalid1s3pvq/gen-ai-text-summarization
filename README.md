# Medical Text Summarization with Entity Preservation

A fine-tuned FLAN-T5 model for medical text summarization that preserves medical entities (conditions, treatments, medications).

## 📁 Project Structure

### Core Files

- **`1data_augment.py`** - Data augmentation with medical entity injection
- **`2fine_tune.py`** - Main fine-tuning script with entity-aware metrics
- **`3evaluate.py`** - Model evaluation with entity retention analysis
- **`4deployment.py`** - Slack bot deployment for medical summarization
- **`api/app.py`** - FastAPI backend that serves summaries via REST
- **`frontend/`** - 3D neon-themed web client for interactive summaries
- **`test_manual.py`** - Interactive testing interface

### Data & Models

- **`augmented_train/`** - Training data with medical entity augmentation
- **`augmented_val/`** - Validation data with medical entities
- **`fine_tuned_model/`** - Fine-tuned FLAN-T5 model with LoRA adapters

### Documentation

- **`ENTITY_METRICS_FIX_SUMMARY.md`** - Detailed analysis of entity metrics implementation
- **`requirements.txt`** - Python dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Manual Testing

```bash
python test_manual.py
```

### 3. Fine-tune Model

```bash
python 2fine_tune.py
```

### 4. Evaluate Model

```bash
python 3evaluate.py
```

### 5. Launch the API

```bash
uvicorn api.app:app --reload
```

### 6. Start the 3D Frontend (Summarize + Orchestrate)

```bash
cd frontend
python -m http.server 5500
# then open http://localhost:5500 in your browser
```

The web UI now lets you:

- Submit medical text for entity-aware summaries
- Kick off fine-tuning jobs (runs `2fine_tune.py`)
- Launch evaluation jobs (runs `3evaluate.py`)
- Monitor live status + log tails for each job within the neon-themed dashboard

## 🔬 Key Features

### Entity-Aware Metrics

- **Entity Retention**: Measures preservation of medical terms from input to summary
- **Entity F1 Score**: Precision and recall of medical entity generation
- **Medical Entity Types**: Conditions (diabetes, cancer) and treatments (insulin, chemotherapy)

### Enhanced Training

- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Medical-focused prompts**: "Summarize this medical text while preserving all medical conditions..."
- **Entity-aware evaluation**: Tracks medical entity preservation during training

### Data Augmentation

- **Medical entity injection**: Adds medical terms to CNN/DailyMail dataset
- **Entity preservation**: Ensures medical terms appear in both articles and summaries
- **51% entity coverage**: Validation data contains medical entities in highlights

## 📊 Model Performance

### Current Metrics

- **ROUGE-L**: 0.22+ (standard summarization quality)
- **BertScore F1**: 0.86+ (semantic similarity)
- **Entity Retention**: Measures medical entity preservation
- **Entity F1**: Precision/recall of medical entity generation

### Training Configuration

- **Model**: FLAN-T5-small with LoRA adapters
- **Training**: 8 epochs, 5e-5 learning rate, cosine scheduler
- **Evaluation**: Entity-aware metrics with medical term tracking

## 🏥 Medical Entity Types

### Conditions

diabetes, hypertension, cancer, asthma, arthritis, epilepsy, anemia

### Treatments/Medications

insulin, metformin, aspirin, chemotherapy, ibuprofen, albuterol

## 💡 Usage Examples

### Medical Text Input

```
"Patient presents with diabetes and hypertension. Prescribed insulin and metformin for glucose control."
```

### Expected Summary

```
"Patient has diabetes and hypertension. Treatment includes insulin and metformin."
```

## 🔧 Technical Details

### Entity Metrics Implementation

The model tracks medical entity preservation using:

1. **Entity Detection**: Identifies medical terms in predictions vs references
2. **Retention Calculation**: Percentage of reference entities preserved
3. **F1 Scoring**: Precision and recall of entity generation

### Training Enhancements

- Medical-focused prompt templates
- Entity-aware loss computation
- Detailed entity analysis during evaluation
- LoRA configuration optimized for medical domain

## 📈 Monitoring

Use Weights & Biases integration to track:

- Training loss and validation metrics
- Entity retention rates during training
- Model performance across epochs
- Detailed entity analysis

---

**Note**: The entity metrics framework is fully functional. Any 0% retention scores indicate the model needs further training to generate medical entities, not metric calculation errors.
