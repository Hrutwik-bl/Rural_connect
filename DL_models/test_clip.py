"""Test CLIP zero-shot classification on our complaint images."""
import open_clip
import torch
from PIL import Image
import os

# Load CLIP model (small, fast version)
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
print("CLIP loaded!")

# Department descriptions for zero-shot classification
dept_descriptions = [
    "a photo of a damaged road with potholes",
    "a photo of water leakage or water supply problem", 
    "a photo of electrical problems with wires or poles"
]
dept_labels = ["Road", "Water", "Electricity"]

# Tokenize text descriptions
text_tokens = tokenizer(dept_descriptions)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def classify_image(img_path):
    """Classify an image using CLIP zero-shot."""
    img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    probs = similarity[0].numpy()
    pred_idx = probs.argmax()
    return dept_labels[pred_idx], float(probs[pred_idx]), {dept_labels[i]: float(probs[i]) for i in range(3)}

# Test on all images
for dept_name in ["road", "water", "electricity"]:
    dept_dir = f"data/images/{dept_name}"
    expected = dept_name.capitalize() if dept_name != "electricity" else "Electricity"
    files = sorted([f for f in os.listdir(dept_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])[:10]
    
    correct = 0
    total = len(files)
    print(f"\n--- {dept_name.upper()} images ---")
    for fname in files:
        pred, conf, all_probs = classify_image(os.path.join(dept_dir, fname))
        is_correct = pred == expected
        correct += int(is_correct)
        mark = "✓" if is_correct else "✗"
        print(f"  {mark} {fname}: {pred} ({conf:.0%}) | R={all_probs['Road']:.0%} W={all_probs['Water']:.0%} E={all_probs['Electricity']:.0%}")
    
    print(f"  Accuracy: {correct}/{total} ({correct/total:.0%})")
