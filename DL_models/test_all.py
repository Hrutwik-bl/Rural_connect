"""
Comprehensive test for the RuralConnect AI API.
Tests all scenarios: matching, mismatching, text-only, rejection.
"""
import requests
import base64
import os
import sys

API_URL = "http://localhost:8003"
BACKEND_URL = "http://localhost:5000"
PASS = 0
FAIL = 0

def test(name, condition, details=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {details}")

def load_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# ============================================
print("=" * 60)
print("RURALCONNECT AI API - COMPREHENSIVE TEST")
print("=" * 60)

# Test 1: Health Check
print("\n--- 1. Health Check ---")
try:
    r = requests.get(f"{API_URL}/", timeout=5).json()
    test("AI API health", r["status"] == "RuralConnect AI API running")
except Exception as e:
    test("AI API health", False, str(e))

try:
    r = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
    test("Backend health", r.status_code == 200)
except Exception as e:
    test("Backend health", False, str(e))

# Test 2: Text-only prediction (no image)
print("\n--- 2. Text-Only Predictions ---")
cases_text = [
    ("road pothole damage", "Road"),
    ("water pipe leaking", "Water"),
    ("electricity power outage", "Electricity"),
]
for desc, expected_dept in cases_text:
    r = requests.post(f"{API_URL}/predict-complaint", json={
        "description": desc, "image_data": ""
    }, timeout=15).json()
    test(f"Text '{desc}' → {expected_dept}",
         r["predicted_department"] == expected_dept,
         f"Got: {r['predicted_department']}")

# Test 3: Image + matching text (VALID)
print("\n--- 3. Image + Matching Text (should be VALID) ---")
img_road = load_image("data/images/road/road_1.jpg")
img_water = load_image("data/images/water/water_1.jpg")
img_elec = load_image("data/images/electricity/electric_1.jpg")

cases_match = [
    (img_road, "road damage pothole", "Road"),
    (img_water, "water pipe leak", "Water"),
    (img_elec, "electricity wire problem", "Electricity"),
]
for img, desc, expected_dept in cases_match:
    r = requests.post(f"{API_URL}/predict-complaint", json={
        "description": desc, "image_data": img
    }, timeout=30).json()
    test(f"Image({expected_dept}) + Text('{desc}') → dept={expected_dept}, valid=True",
         r["predicted_department"] == expected_dept and r["validity"]["is_valid"] == True,
         f"Got: dept={r['predicted_department']}, valid={r['validity']['is_valid']}")

# Test 4: Image + MISMATCHING text (should be NOT VALID)
print("\n--- 4. Image + Mismatching Text (should be NOT VALID) ---")
cases_mismatch = [
    (img_road, "water leakage in pipes", "Road", "Water"),   # Road image + water text
    (img_road, "electricity problem with wires", "Road", "Electricity"),  # Road image + electricity text
    (img_water, "road damage pothole", "Water", "Road"),     # Water image + road text
    (img_elec, "water supply issue", "Electricity", "Water"),  # Electricity image + water text
]
for img, desc, img_dept, text_dept in cases_mismatch:
    r = requests.post(f"{API_URL}/predict-complaint", json={
        "description": desc, "image_data": img
    }, timeout=30).json()
    test(f"Image({img_dept}) + Text('{desc}') → dept={img_dept}, valid=False",
         r["predicted_department"] == img_dept and r["validity"]["is_valid"] == False,
         f"Got: dept={r['predicted_department']}, valid={r['validity']['is_valid']}")

# Test 5: Rejection (not a complaint)
print("\n--- 5. Non-Complaint Rejection ---")
non_complaints = [
    "hello how are you",
    "good morning have a nice day",
]
for desc in non_complaints:
    r = requests.post(f"{API_URL}/predict-complaint", json={
        "description": desc, "image_data": ""
    }, timeout=15).json()
    test(f"'{desc}' → rejected",
         r.get("rejected") == True,
         f"Got: rejected={r.get('rejected')}, dept={r.get('predicted_department')}")

# Test 6: Severity predictions
print("\n--- 6. Severity Predictions ---")
severity_cases = [
    ("there is a small crack on the road", ["Low", "Medium"]),
    ("dangerous pothole causing accidents frequently", ["High", "Critical"]),
    ("urgent emergency road collapsed", ["Critical"]),
]
for desc, expected_sevs in severity_cases:
    r = requests.post(f"{API_URL}/predict-complaint", json={
        "description": desc, "image_data": ""
    }, timeout=15).json()
    test(f"'{desc[:40]}...' → sev in {expected_sevs}",
         r["predicted_severity"] in expected_sevs,
         f"Got: {r['predicted_severity']}")

# Test 7: Backend predict endpoint
print("\n--- 7. Backend /api/complaints/predict ---")
try:
    # Login first
    login_r = requests.post(f"{BACKEND_URL}/api/auth/login", json={
        "email": "testrunner@test.com",
        "password": "Test@1234"
    }, timeout=10)
    if login_r.status_code == 200:
        token = login_r.json().get("token")
        test("Login successful", token is not None)
        
        # Test predict through backend
        predict_r = requests.post(f"{BACKEND_URL}/api/complaints/predict", 
            json={"description": "road has big potholes", "imageData": img_road},
            headers={"Authorization": f"Bearer {token}"},
            timeout=120
        )
        pr = predict_r.json()
        test("Backend predict → Road",
             pr.get("predicted_department") == "Road",
             f"Got: {pr.get('predicted_department', 'ERROR')}")
        test("Backend predict → valid=True (matching)",
             pr.get("validity", {}).get("is_valid") == True,
             f"Got: {pr.get('validity', {}).get('is_valid')}")
    else:
        test("Login", False, f"Status: {login_r.status_code}")
except Exception as e:
    test("Backend predict", False, str(e))

# Test 8: Location verification
print("\n--- 8. Location Verification ---")
try:
    r = requests.post(f"{API_URL}/verify-location", json={
        "complaint_lat": 13.0827,
        "complaint_lon": 77.5877,
        "resolved_lat": 13.0830,
        "resolved_lon": 77.5880
    }, timeout=10).json()
    test("Location verify returns distance",
         "distance_meters" in r and r["distance_meters"] < 500,
         f"Got: distance={r.get('distance_meters')}")
except Exception as e:
    test("Location verify", False, str(e))

# Summary
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print("=" * 60)
if FAIL == 0:
    print("🎉 ALL TESTS PASSED!")
else:
    print(f"⚠ {FAIL} test(s) failed")
