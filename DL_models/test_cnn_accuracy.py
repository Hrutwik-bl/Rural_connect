import requests, base64, os, glob

# Test all road images to see what CNN predicts
img_dir = "data/images/road/"
files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and 'augmented' not in f])

print(f"Testing {len(files)} road images with neutral text 'issue reported'...\n")

for fname in files:
    img_path = img_dir + fname
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    resp = requests.post("http://localhost:8003/predict-complaint", json={
        "description": "issue reported",
        "image_data": f"data:image/jpeg;base64,{img_b64}"
    }, timeout=30)
    r = resp.json()
    print(f"{fname}: Dept={r['predicted_department']} (conf={r['department_confidence']}) | Sev={r['predicted_severity']}")

# Also test water images
print("\n--- Water images ---")
water_dir = "data/images/water/"
water_files = sorted([f for f in os.listdir(water_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and 'augmented' not in f])[:5]
for fname in water_files:
    img_path = water_dir + fname
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = requests.post("http://localhost:8003/predict-complaint", json={
        "description": "issue reported",
        "image_data": f"data:image/jpeg;base64,{img_b64}"
    }, timeout=30)
    r = resp.json()
    print(f"{fname}: Dept={r['predicted_department']} (conf={r['department_confidence']}) | Sev={r['predicted_severity']}")

# Also test electricity images
print("\n--- Electricity images ---")
elec_dir = "data/images/electricity/"
elec_files = sorted([f for f in os.listdir(elec_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and 'augmented' not in f])[:5]
for fname in elec_files:
    img_path = elec_dir + fname
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = requests.post("http://localhost:8003/predict-complaint", json={
        "description": "issue reported",
        "image_data": f"data:image/jpeg;base64,{img_b64}"
    }, timeout=30)
    r = resp.json()
    print(f"{fname}: Dept={r['predicted_department']} (conf={r['department_confidence']}) | Sev={r['predicted_severity']}")
