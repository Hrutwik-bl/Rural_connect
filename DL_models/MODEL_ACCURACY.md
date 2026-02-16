# Model Accuracy Report

## Date: January 31, 2026

### 1. CNN Image Classification Model
- **Validation Accuracy:** 83.33%
- **Dataset:** 60 images (20 per department)
- **Split:** 80% train, 20% validation
- **Model:** MobileNetV2 transfer learning

### 2. RNN Text Classification Model
- **Test Accuracy:** 85.00%
- **Dataset:** Text complaints from complaints.csv
- **Split:** 80% train, 20% test (stratified)
- **Model:** Bidirectional LSTM

### 3. Location Verification Model
- **Test Accuracy:** 100.00%
- **Dataset:** GPS coordinates from location_verification.csv
- **Split:** 80% train, 20% test (stratified)
- **Model:** Dense Neural Network (Haversine distance-based)

### 4. Multimodal Model (CNN + LSTM)
- **Department Prediction Accuracy:** 40.00%
- **Severity Prediction Accuracy:** 37.78%
- **Validation Score Accuracy:** 53.33%
- **Dataset:** 45 samples with image + text pairs
- **Model:** Custom CNN + LSTM with cosine similarity validation
- **Note:** Trained from scratch, not using pretrained weights

---

## Summary
- **Best performing:** Location Verification (100%)
- **Text classification:** RNN (85%)
- **Image classification:** CNN (83.33%)
- **Multimodal fusion:** Lower accuracy due to small dataset size (45 samples)

## Recommendations
- Increase multimodal dataset size for better accuracy
- Consider data augmentation for multimodal training
- Fine-tune hyperparameters for multimodal model
