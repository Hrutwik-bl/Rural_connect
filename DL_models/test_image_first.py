import requests, base64, os

# Use a road image from training data
img_dir = "data/images/road/"
files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Available road images: {files[:5]}")
img_path = img_dir + files[0]
print(f"Using: {img_path}")

with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Test 1: Road image + "electricity leakage" text
resp = requests.post("http://localhost:8003/predict-complaint", json={
    "description": "electricity leakage",
    "image_data": f"data:image/jpeg;base64,{img_b64}"
}, timeout=30)
r = resp.json()
print("\n=== Test 1: Road image + 'electricity leakage' text ===")
print(f"Department: {r['predicted_department']}")
print(f"Dept Confidence: {r['department_confidence']}")
print(f"Severity: {r['predicted_severity']}")
print(f"Method: {r.get('method', 'N/A')}")
print(f"CNN image dept: {r.get('cnn_image_department', 'N/A')}")
print(f"GPT-2 text dept: {r.get('gpt2_department', 'N/A')}")

# Test 2: Road image + "water leakage" text
resp2 = requests.post("http://localhost:8003/predict-complaint", json={
    "description": "water leakage on road",
    "image_data": f"data:image/jpeg;base64,{img_b64}"
}, timeout=30)
r2 = resp2.json()
print("\n=== Test 2: Road image + 'water leakage on road' text ===")
print(f"Department: {r2['predicted_department']}")
print(f"Dept Confidence: {r2['department_confidence']}")
print(f"CNN image dept: {r2.get('cnn_image_department', 'N/A')}")
print(f"GPT-2 text dept: {r2.get('gpt2_department', 'N/A')}")

# Test 3: Road image + "road damage" text (should agree)
resp3 = requests.post("http://localhost:8003/predict-complaint", json={
    "description": "road damage potholes",
    "image_data": f"data:image/jpeg;base64,{img_b64}"
}, timeout=30)
r3 = resp3.json()
print("\n=== Test 3: Road image + 'road damage potholes' text ===")
print(f"Department: {r3['predicted_department']}")
print(f"Dept Confidence: {r3['department_confidence']}")
print(f"CNN image dept: {r3.get('cnn_image_department', 'N/A')}")
print(f"GPT-2 text dept: {r3.get('gpt2_department', 'N/A')}")
