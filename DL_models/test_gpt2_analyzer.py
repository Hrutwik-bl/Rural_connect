"""Test script for GPT2TextAnalyzer"""
import sys
sys.path.insert(0, '.')
from gpt2_analyzer import GPT2TextAnalyzer

analyzer = GPT2TextAnalyzer()
print("Loading models...")
loaded = analyzer.load_models()
print(f"Models loaded: {loaded}\n")

# Test cases
test_texts = [
    "There is a huge pothole on the main road causing accidents",
    "Water pipeline burst near the school area",
    "No electricity in our village since 3 days",
    "The road is in good condition, no problems at all",
    "Everything is working fine, no issues",
    "Transformer exploded and wires are hanging dangerously",
    "Drainage overflow is flooding the entire street",
]

print("=" * 70)
print("TESTING GPT-2 TEXT ANALYZER")
print("=" * 70)

for text in test_texts:
    print(f'\nInput: "{text}"')
    result = analyzer.analyze_text(text)
    sentiment = result.get("sentiment", {})
    dept = result.get("department", {})
    print(f'  Valid Complaint: {result["is_valid_complaint"]}')
    print(f'  Sentiment: {sentiment.get("label", "N/A")} (conf: {sentiment.get("confidence", 0):.2%})')
    if result["is_valid_complaint"] and dept:
        print(f'  Department: {dept.get("department", "N/A")} (conf: {dept.get("confidence", 0):.2%})')
    elif not result["is_valid_complaint"]:
        print(f'  Rejection: {result.get("rejection_reason", "N/A")}')
    print("-" * 70)

# Test cross-validation
print("\n" + "=" * 70)
print("TESTING CROSS-VALIDATION WITH CNN")
print("=" * 70)
text = "Water leaking from a broken pipe near the park"
cnn_dept = "Water"
cnn_conf = 0.85
cv = analyzer.cross_validate_with_cnn(text, cnn_dept, cnn_conf)
print(f'Text: "{text}"')
print(f"CNN prediction: {cnn_dept} ({cnn_conf:.0%})")
print(f'Agreement: {cv["agreement"]}')
print(f'Final department: {cv["final_department"]} ({cv["final_confidence"]:.2%})')
print(f'Message: {cv["message"]}')

# Test disagreement case
print("\n" + "=" * 70)
print("TESTING CROSS-VALIDATION DISAGREEMENT")
print("=" * 70)
text2 = "There is a big pothole on MG Road"
cnn_dept2 = "Water"
cnn_conf2 = 0.60
cv2 = analyzer.cross_validate_with_cnn(text2, cnn_dept2, cnn_conf2)
print(f'Text: "{text2}"')
print(f"CNN prediction: {cnn_dept2} ({cnn_conf2:.0%})")
print(f'Agreement: {cv2["agreement"]}')
print(f'Final department: {cv2["final_department"]} ({cv2["final_confidence"]:.2%})')
print(f'Message: {cv2["message"]}')

print("\n\nAll tests completed!")
