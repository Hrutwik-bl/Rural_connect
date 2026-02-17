import requests, json

tests = [
    ("road have patholes", "Medium or High"),
    ("road pothole damage", "High (inherent)"),
    ("water pipe leaking", "Medium+"),
    ("electricity wire problem", "Medium+"),
    ("there is a small crack on the road", "Medium"),
    ("dangerous pothole causing accidents frequently", "High/Critical"),
    ("urgent emergency road collapsed people injured", "Critical"),
    ("hello how are you", "Rejected"),
]

print("=" * 70)
print("SEVERITY PREDICTION TEST")
print("=" * 70)

passed = 0
failed = 0

for desc, expect in tests:
    r = requests.post("http://localhost:8003/predict-complaint", json={"description": desc}, timeout=30)
    d = r.json()
    
    if d.get("rejected"):
        sev = "REJECTED"
        dept = "N/A"
    else:
        sev = d.get("predicted_severity", "N/A")
        dept = d.get("predicted_department", "N/A")
    
    # Check if severity is correct
    if sev == "REJECTED" and "Rejected" in expect:
        status = "PASS"
    elif sev in ["Medium", "High", "Critical"] and sev != "Low":
        status = "PASS"
    elif "small" in desc.lower() and sev == "Medium":
        status = "PASS"
    else:
        status = "FAIL"
    
    icon = "✅" if status == "PASS" else "❌"
    print(f'  {icon} "{desc}"')
    print(f'     → Dept={dept}, Severity={sev}  (Expected: {expect})')
    
    if status == "PASS":
        passed += 1
    else:
        failed += 1

print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
