"""
Create balanced dataset with 50% valid (matching) and 50% invalid (mismatching) pairs
"""
import pandas as pd
import os
import random

# Set random seed for reproducibility
random.seed(42)

# Define complaints for each department
water_complaints = [
    "There is water leakage from the pipe near my house",
    "Pipe burst in our village, water is overflowing",
    "Drainage system is blocked, causing water stagnation",
    "Water supply is contaminated with dirt and sediment",
    "Broken water tap causing wastage in the village",
    "Water pipeline needs urgent repair",
    "Sewage water is mixing with drinking water supply",
    "Water tank overflow creating flooding in the area",
    "Leakage in underground water pipe near market",
    "Water supply interrupted for 3 days",
    "Pipe leakage near my village main road",
    "Water is not available for past week",
    "Rusty water coming from supply line",
    "Damage to water distribution network",
    "Emergency water supply issue in residential area",
]

road_complaints = [
    "Large potholes on main village road causing accidents",
    "Broken road near government school needs repair",
    "Cracks on road after heavy rainfall",
    "Uneven road surface near market area",
    "Street lighting on damaged road is not working",
    "Road damage causing traffic congestion",
    "Dangerous pothole near bus stop",
    "Road needs immediate repair and patching",
    "Broken asphalt on village main road",
    "Cracked pavement near residential area",
    "Pothole creating hazard for vehicles",
    "Road deterioration in poor condition",
    "Damaged street pavement needs reconstruction",
    "Highway pothole requires urgent fixing",
    "Village road full of cracks and damage",
]

electricity_complaints = [
    "Electricity pole is damaged and leaning dangerously",
    "Power wire is broken and hanging near road",
    "Electricity transformer needs repair",
    "Street light is not working for weeks",
    "Power cut happening frequently in village",
    "Electrical wire creating safety hazard",
    "High voltage wire hanging low near school",
    "Electricity supply is unstable causing flickering",
    "Power pole maintenance required urgently",
    "Faulty electrical connection near residential area",
    "Electric wire short circuit needs attention",
    "Electricity meter showing wrong reading",
    "Power lines need immediate repair",
    "Transformer station requires maintenance",
    "Electrical fault causing frequent outages",
]

# Get available images
base_path = "data/images"
water_images = [f"water/{f}" for f in os.listdir(f"{base_path}/water") if f.endswith('.jpg')][:15]
road_images = [f"road/{f}" for f in os.listdir(f"{base_path}/road") if f.endswith('.jpg')][:15]
electricity_images = [f"electricity/{f}" for f in os.listdir(f"{base_path}/electricity") if f.endswith('.jpg')][:15]

print(f"Available water images: {len(water_images)}")
print(f"Available road images: {len(road_images)}")
print(f"Available electricity images: {len(electricity_images)}")

# Create balanced dataset
data = []

# WATER - Valid pairs (matching)
for i, img in enumerate(water_images[:8]):
    data.append({
        'description': water_complaints[i % len(water_complaints)],
        'image_path': img,
        'department': 'Water',
        'severity': random.choice(['Low', 'Medium', 'High', 'Critical']),
        'label': 'water',
        'is_valid': 1  # Valid - matching pair
    })

# WATER - Invalid pairs (mismatching - road images with water description)
for i, img in enumerate(road_images[:7]):
    data.append({
        'description': water_complaints[i % len(water_complaints)],
        'image_path': img,
        'department': 'Water',
        'severity': random.choice(['Low', 'Medium', 'High', 'Critical']),
        'label': 'water',
        'is_valid': 0  # Invalid - mismatched pair (water text with road image)
    })

# ROAD - Valid pairs (matching)
for i, img in enumerate(road_images[:8]):
    data.append({
        'description': road_complaints[i % len(road_complaints)],
        'image_path': img,
        'department': 'Road',
        'severity': random.choice(['Low', 'Medium', 'High', 'Critical']),
        'label': 'road',
        'is_valid': 1  # Valid - matching pair
    })

# ROAD - Invalid pairs (mismatching - electricity images with road description)
for i, img in enumerate(electricity_images[:7]):
    data.append({
        'description': road_complaints[i % len(road_complaints)],
        'image_path': img,
        'department': 'Road',
        'severity': random.choice(['Low', 'Medium', 'High', 'Critical']),
        'label': 'road',
        'is_valid': 0  # Invalid - mismatched pair
    })

# ELECTRICITY - Valid pairs (matching)
for i, img in enumerate(electricity_images[:8]):
    data.append({
        'description': electricity_complaints[i % len(electricity_complaints)],
        'image_path': img,
        'department': 'Electricity',
        'severity': random.choice(['Low', 'Medium', 'High', 'Critical']),
        'label': 'electricity',
        'is_valid': 1  # Valid - matching pair
    })

# ELECTRICITY - Invalid pairs (mismatching - water images with electricity description)
for i, img in enumerate(water_images[8:15]):
    data.append({
        'description': electricity_complaints[i % len(electricity_complaints)],
        'image_path': img,
        'department': 'Electricity',
        'severity': random.choice(['Low', 'Medium', 'High', 'Critical']),
        'label': 'electricity',
        'is_valid': 0  # Invalid - mismatched pair
    })

# Create DataFrame and shuffle
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
csv_path = 'data/text/complaints_with_valid.csv'
df.to_csv(csv_path, index=False)

print(f"\n✅ Dataset created with {len(df)} samples")
print(f"Valid pairs: {(df['is_valid'] == 1).sum()}")
print(f"Invalid pairs: {(df['is_valid'] == 0).sum()}")
print(f"\nDataset distribution by department:")
print(df['department'].value_counts())
print(f"\nSaved to: {csv_path}")
