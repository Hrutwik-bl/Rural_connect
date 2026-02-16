"""
Text augmentation for limited complaint dataset
Generates variations of complaint descriptions using NLP techniques
"""

import pandas as pd
import random
from pathlib import Path

# Synonyms and variations for common complaint terms
SYNONYMS = {
    'water': ['water', 'water supply', 'tap water', 'drinking water'],
    'leak': ['leak', 'leakage', 'leaking', 'seeping', 'dripping'],
    'pipe': ['pipe', 'pipeline', 'water line', 'main', 'pipe line'],
    'broken': ['broken', 'damaged', 'ruptured', 'burst', 'fractured'],
    'road': ['road', 'street', 'highway', 'pavement', 'asphalt'],
    'pothole': ['pothole', 'pit', 'crater', 'hole', 'depression', 'cavity'],
    'crack': ['crack', 'fissure', 'fracture', 'split', 'crevice'],
    'electric': ['electric', 'electrical', 'power', 'electricity'],
    'wire': ['wire', 'cable', 'line', 'power line', 'electric line'],
    'pole': ['pole', 'post', 'electric pole', 'power pole'],
    'issue': ['issue', 'problem', 'fault', 'defect', 'damage'],
    'need': ['need', 'require', 'require immediate', 'urgent'],
    'repair': ['repair', 'fix', 'maintenance', 'restoration'],
    'near': ['near', 'at', 'close to', 'by', 'around'],
}

# Template variations
TEMPLATES = [
    "{issue} with {main} {location}",
    "{main} is {status} {location}",
    "{issue} in {main} near {location}",
    "There is {article} {adjective} {main} causing {consequence}",
    "{main} {status} at {location}",
    "Urgent: {main} {status} {location}",
]

ARTICLES = ['a', 'the', 'an']
ADJECTIVES = ['severe', 'major', 'critical', 'urgent', 'ongoing', 'continuous']
CONSEQUENCES = ['water wastage', 'flood risk', 'traffic hazard', 'safety concern', 'inconvenience']


def augment_text(original_text, num_variations=5):
    """Generate variations of a complaint text"""
    variations = [original_text]
    
    for _ in range(num_variations):
        text = original_text.lower()
        
        # Replace synonyms randomly
        for word, synonyms in SYNONYMS.items():
            if word in text:
                replacement = random.choice(synonyms)
                text = text.replace(word, replacement)
        
        # Add articles, adjectives, or restructure
        if random.random() > 0.5 and len(text.split()) > 5:
            # Add urgency
            text = f"Urgent: {text}"
        
        if text != original_text and text not in variations:
            variations.append(text)
    
    return variations[:num_variations + 1]


def load_and_augment_dataset(csv_path, output_path, augmentations_per_sample=5):
    """Load dataset and create augmented version"""
    
    df = pd.read_csv(csv_path)
    
    augmented_rows = []
    
    for idx, row in df.iterrows():
        original_desc = row['description']
        department = row['department']
        severity = row.get('severity', 'Medium')
        
        # Add original
        augmented_rows.append({
            'description': original_desc,
            'department': department,
            'severity': severity,
            'is_augmented': 0
        })
        
        # Generate augmented variations
        variations = augment_text(original_desc, num_variations=augmentations_per_sample)
        
        for variation in variations[1:]:  # Skip first (original)
            augmented_rows.append({
                'description': variation,
                'department': department,
                'severity': severity,
                'is_augmented': 1
            })
    
    augmented_df = pd.DataFrame(augmented_rows)
    
    # Save augmented dataset
    augmented_df.to_csv(output_path, index=False)
    
    print(f"✅ Original samples: {len(df)}")
    print(f"✅ Augmented samples: {len(augmented_df)}")
    print(f"✅ Saved to: {output_path}")
    
    return augmented_df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    original_csv = BASE_DIR / "data" / "text" / "complaints_with_valid.csv"
    augmented_csv = BASE_DIR / "data" / "text" / "complaints_augmented.csv"
    
    if original_csv.exists():
        augmented_df = load_and_augment_dataset(
            original_csv,
            augmented_csv,
            augmentations_per_sample=4
        )
        print("\n📊 Sample augmented descriptions:")
        print(augmented_df[augmented_df['is_augmented'] == 1].head(10).to_string())
    else:
        print(f"❌ File not found: {original_csv}")
