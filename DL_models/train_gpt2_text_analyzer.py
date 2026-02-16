"""
GPT-2 Text Analyzer Training Script
====================================
Fine-tunes GPT-2 for two tasks:
  1. Sentiment Classification: Detects if text is a negative complaint or not
  2. Department Classification: Classifies complaint into Electricity/Road/Water
  
Uses eos_token as pad_token with LEFT padding (standard GPT-2 approach).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup
)

# =====================================================
# CONFIGURATION
# =====================================================
SEED = 42
MAX_LEN = 64
BATCH_SIZE = 8
SENTIMENT_EPOCHS = 8
DEPT_EPOCHS = 10
LEARNING_RATE = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")

# =====================================================
# NON-COMPLAINT (POSITIVE/NEUTRAL) TEXT EXAMPLES
# =====================================================
NON_COMPLAINT_TEXTS = [
    "The road in our village has been repaired beautifully",
    "Thank you for fixing the water supply so quickly",
    "The electricity is working perfectly now",
    "Great job on the new road construction",
    "Water supply has improved significantly this month",
    "The new street lights are working wonderfully",
    "Power supply has been very stable recently",
    "The drainage system is clean and well maintained",
    "Road widening project completed successfully",
    "Transformer replacement done on time thank you",
    "Village roads are in excellent condition after repair",
    "Water quality has improved after the new filter",
    "Electricity connections have been properly restored",
    "The potholes were filled promptly and effectively",
    "Street lighting project is beneficial for safety",
    "What are the water supply timings in our area",
    "When will the new road construction start",
    "Information about electricity tariff rates",
    "How to apply for new water connection",
    "Details about road development plan for village",
    "Schedule of power maintenance in our area",
    "How many hours is water supplied daily",
    "Request for information about village infrastructure",
    "What is the contact number for electricity department",
    "Where is the nearest water testing lab",
    "The weather is nice today in our village",
    "There is a festival celebration next week",
    "Schools will reopen after summer holidays",
    "The market will be closed on Sunday",
    "New hospital inaugurated in the town",
    "Farming season has started with good rain",
    "Village meeting scheduled for next Monday",
    "Cricket tournament organized for youth",
    "Health camp organized in the village",
    "Vaccination drive completed successfully",
    "Trees planted along the village boundary",
    "Community hall renovation looking good",
    "Good harvest expected this season",
    "Local transport service improved recently",
    "Internet connectivity available in village now",
    "I appreciate the quick response from water department",
    "Excellent work on restoring power after the storm",
    "The road repair crew did a fantastic job",
    "Water department resolved our issue within hours",
    "Very happy with the new drainage system installed",
    "Street lights make our village feel safe at night",
    "The new pipeline brings clean water to our homes",
    "All services running smoothly in our village",
    "Good progress on infrastructure development",
    "Happy with the quick repair of street lights",
    "Water tanker arrived on time today",
    "Power restored within one hour of complaint",
    "Road repair completed ahead of schedule",
    "New connections provided to all houses",
    "Water pressure is perfect now after repairs",
    "Electricity bill reduced after meter replacement",
    "Safe road crossings installed near school",
    # ========== NEGATION EXAMPLES ==========
    # These look like complaints but negate the problem (NOT complaints)
    "no water leakage",
    "no water leakage in our area",
    "no water leakage from pipes",
    "there is no water leakage",
    "no leakage found anywhere",
    "no pothole on the road",
    "no potholes in our village",
    "there are no potholes here",
    "no road damage at all",
    "no road damage in our area",
    "there is no road damage",
    "no electricity problem",
    "no electricity issues",
    "no power outage today",
    "no power cut in our area",
    "there is no electricity problem",
    "no water problem",
    "no water issue in our area",
    "no drainage issue",
    "no drainage overflow",
    "there is no drainage problem",
    "no broken pipe",
    "no broken road",
    "no broken street light",
    "not any water leakage",
    "not any pothole on road",
    "not any electricity issue",
    "water is not leaking",
    "road is not damaged",
    "electricity is not disrupted",
    "power is not cut",
    "water is not dirty",
    "pipe is not broken",
    "road is not broken",
    "no issue with water supply",
    "no issue with electricity",
    "no issue with road",
    "no problem with water",
    "no problem with road",
    "no problem with electricity",
    "no complaint about water",
    "no complaint about road",
    "no complaint about electricity",
    "there is no problem here",
    "everything is fine with water",
    "everything is fine with electricity",
    "everything is fine with road",
    "nothing wrong with the road",
    "nothing wrong with water supply",
    "nothing wrong with electricity",
    "water supply is working fine",
    "road is in good condition",
    "electricity is working properly",
    "no damage to the road",
    "no damage to water pipes",
    "no damage to power lines",
    "there is no flooding",
    "no flooding in our area",
    "no water shortage",
    "no power failure",
    "no transformer issue",
    "no water contamination",
    "no road accident here",
    "road is fine no issues",
    "water supply is okay",
    "electricity supply is stable",
    "no cracks on road",
    "no pipe burst",
    "no sewage overflow",
    "no wire hanging",
    "everything works fine",
    "all is good",
    "nothing is broken",
    "nothing to complain about",
    "no issues found",
    "no problems reported",
    "services are running fine",
    "no disruption in power supply",
    "no disruption in water supply",
    "water is fine no problem",
    "road is fine no damage",
    "power is stable no cuts",
    "no water supply problem",
    "no electricity supply issue",
    "nothing broken on the road",
    "there is no crack on road",
    "drainage is not blocked",
    "water pressure is normal",
    "no low pressure issue",
    "road surface is smooth",
    "no accident has happened",
    "everything is normal",
    "all fine no complaint",
    "no need to complain",
    "no complaints from our side",
    "there is no issue",
    "not facing any problem",
    "no water logging",
    "no street light issue",
    "no electric pole damage",
]


def setup_tokenizer():
    """Setup GPT-2 tokenizer: use eos_token as pad, LEFT padding"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_len,
            padding='max_length', return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


def load_complaint_data():
    csv_path = DATA_DIR / "text" / "complaints_augmented.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "text" / "complaints.csv"
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} complaint records from {csv_path.name}")
    return df


def prepare_sentiment_data():
    print("\n  Preparing SENTIMENT data...")
    df = load_complaint_data()
    complaint_texts = df['description'].tolist()
    complaint_labels = [1] * len(complaint_texts)

    non_complaint_texts = NON_COMPLAINT_TEXTS.copy()
    non_complaint_labels = [0] * len(non_complaint_texts)

    # Augment non-complaints to balance
    while len(non_complaint_texts) < len(complaint_texts):
        base = random.choice(NON_COMPLAINT_TEXTS)
        variations = [
            base.lower(),
            "Update: " + base,
            base + " in our area",
            base.replace("village", "town"),
            base.replace("our", "the"),
        ]
        for v in variations:
            if len(non_complaint_texts) < len(complaint_texts):
                non_complaint_texts.append(v)
                non_complaint_labels.append(0)

    all_texts = complaint_texts + non_complaint_texts
    all_labels = complaint_labels + non_complaint_labels
    print(f"  Complaints: {sum(all_labels)}, Non-complaints: {len(all_labels) - sum(all_labels)}")
    return all_texts, all_labels


def prepare_department_data():
    print("\n  Preparing DEPARTMENT data...")
    df = load_complaint_data()
    dept_map = {'Electricity': 0, 'Road': 1, 'Water': 2}
    texts = df['description'].tolist()
    labels = [dept_map.get(d, 1) for d in df['department'].tolist()]
    from collections import Counter
    counts = Counter(labels)
    print(f"  Electricity: {counts[0]}, Road: {counts[1]}, Water: {counts[2]}")
    return texts, labels


def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, task_name):
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    print(f"\n  Best {task_name} Val Accuracy: {best_val_acc:.3f}")
    print(classification_report(all_labels_list, all_preds, digits=3))
    return model, best_val_acc


def train_sentiment():
    print("\n" + "="*60)
    print("  TRAINING GPT-2 SENTIMENT MODEL")
    print("="*60)

    tokenizer = setup_tokenizer()
    texts, labels = prepare_sentiment_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_ds = TextClassificationDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id  # = eos_token_id = 50256
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * SENTIMENT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    model, best_acc = train_model(model, train_loader, val_loader, optimizer, scheduler, SENTIMENT_EPOCHS, "Sentiment")

    save_dir = MODELS_DIR / "gpt2_sentiment"
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # Quick verification
    print("\n  --- Quick Verification ---")
    test_texts = [
        "water leakage from pipe",
        "road is damaged badly",
        "no electricity since morning",
        "The road has been repaired nicely",
        "Thank you for good service",
        "no water leakage",
        "no pothole on the road",
        "no electricity problem",
        "there is no water leakage",
        "road is not damaged",
        "water is not leaking",
        "no issue with water supply",
        "everything is fine with road",
        "nothing wrong with electricity",
        "no complaints from our side",
    ]
    model.eval()
    for t in test_texts:
        enc = tokenizer(t, truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            out = model(input_ids=enc['input_ids'].to(DEVICE), attention_mask=enc['attention_mask'].to(DEVICE))
        p = torch.softmax(out.logits, dim=-1)[0]
        lbl = "COMPLAINT" if torch.argmax(p).item() == 1 else "NOT_COMPLAINT"
        print(f"    '{t}' → {lbl} ({p[1].item():.1%} complaint)")

    with open(save_dir / "training_config.json", "w") as f:
        json.dump({
            "task": "sentiment_classification",
            "labels": {"0": "not_complaint", "1": "complaint"},
            "best_accuracy": round(best_acc, 4),
            "max_len": MAX_LEN, "model_type": "gpt2",
            "pad_token": "eos_token", "padding_side": "left"
        }, f, indent=2)

    print(f"\n  Saved to {save_dir}")
    return best_acc


def train_department():
    print("\n" + "="*60)
    print("  TRAINING GPT-2 DEPARTMENT MODEL")
    print("="*60)

    tokenizer = setup_tokenizer()
    texts, labels = prepare_department_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_ds = TextClassificationDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * DEPT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    model, best_acc = train_model(model, train_loader, val_loader, optimizer, scheduler, DEPT_EPOCHS, "Department")

    save_dir = MODELS_DIR / "gpt2_department"
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    with open(save_dir / "training_config.json", "w") as f:
        json.dump({
            "task": "department_classification",
            "labels": {"0": "Electricity", "1": "Road", "2": "Water"},
            "best_accuracy": round(best_acc, 4),
            "max_len": MAX_LEN, "model_type": "gpt2",
            "pad_token": "eos_token", "padding_side": "left"
        }, f, indent=2)

    print(f"\n  Saved to {save_dir}")
    return best_acc


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  GPT-2 TEXT ANALYZER TRAINING")
    print("="*60)

    s_acc = train_sentiment()
    d_acc = train_department()

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Sentiment Accuracy:  {s_acc:.3f}")
    print(f"  Department Accuracy: {d_acc:.3f}")
    print(f"  Models: {MODELS_DIR}/gpt2_sentiment/ & gpt2_department/")
