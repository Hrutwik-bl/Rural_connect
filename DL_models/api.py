from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import math
import json
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import open_clip

# Import GPT-2 Text Analyzer
from gpt2_analyzer import GPT2TextAnalyzer

app = FastAPI(title="RuralConnect AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CONFIGURATION
# =====================================================
MAX_LEN = 50
IMG_SIZE = (224, 224)

# =====================================================
# LOAD LOCATION VERIFICATION MODEL
# =====================================================
location_verification_model = tf.keras.models.load_model(
    "models/location_verification_model.h5"
)
print("✅ Location verification model loaded")

# =====================================================
# CUSTOM FUNCTIONS FOR MODEL LOADING
# =====================================================
def normalize(x):
    """Normalize features for cosine similarity"""
    return x / (tf.norm(x, axis=-1, keepdims=True) + 1e-7)

def cosine_similarity(x):
    """Compute cosine similarity between image and text features"""
    img, txt = x
    return tf.reduce_sum(img * txt, axis=-1, keepdims=True)

# =====================================================
# LOAD MULTIMODAL MODEL
# =====================================================
try:
    multimodal_model = tf.keras.models.load_model(
        "models/multimodal_model_transfer.h5",
        custom_objects={
            'normalize': normalize,
            'cosine_similarity': cosine_similarity
        }
    )
    print("✅ Multimodal model loaded (transfer learning)")
    MULTIMODAL_AVAILABLE = True
except Exception as e:
    print(f"⚠ Could not load transfer model: {e}")
    # Try loading the simpler model
    try:
        multimodal_model = tf.keras.models.load_model(
            "models/multimodal_model.h5",
            custom_objects={
                'normalize': normalize,
                'cosine_similarity': cosine_similarity
            }
        )
        print("✅ Multimodal model loaded (simple)")
        MULTIMODAL_AVAILABLE = True
    except Exception as e2:
        print(f"⚠ Could not load multimodal model: {e2}")
        print("⚠ Falling back to keyword-based prediction")
        multimodal_model = None
        MULTIMODAL_AVAILABLE = False

# =====================================================
# LOAD TOKENIZER AND ENCODERS
# =====================================================
with open("models/tokenizer_transfer.json", "r") as f:
    tokenizer_data = json.load(f)
    word_index = tokenizer_data["word_index"]

with open("models/dept_encoder_transfer.json", "r") as f:
    dept_classes = json.load(f)["classes"]

with open("models/sev_encoder_transfer.json", "r") as f:
    sev_classes = json.load(f)["classes"]

print(f"✅ Tokenizer loaded (vocab: {len(word_index)} words)")
print(f"✅ Departments: {dept_classes}")
print(f"✅ Severities: {sev_classes}")

# =====================================================
# LOAD GPT-2 TEXT ANALYZER
# =====================================================
gpt2_analyzer = GPT2TextAnalyzer()
GPT2_AVAILABLE = gpt2_analyzer.load_models()
print(f"{'✅' if GPT2_AVAILABLE else '⚠'} GPT-2 Text Analyzer: {'loaded' if GPT2_AVAILABLE else 'not available'}")

# =====================================================
# LOAD CLIP MODEL FOR IMAGE DEPARTMENT CLASSIFICATION
# =====================================================
CLIP_AVAILABLE = False
clip_model = None
clip_preprocess = None
clip_text_features = None
CLIP_DEPARTMENTS = ["Road", "Water", "Electricity"]
CLIP_DESCRIPTIONS = [
    "a photo of a damaged road with potholes",
    "a photo of water leakage or water supply problem",
    "a photo of electrical problems with wires or poles"
]

try:
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    
    # Pre-compute text features for department descriptions
    text_tokens = clip_tokenizer(CLIP_DESCRIPTIONS)
    with torch.no_grad():
        clip_text_features = clip_model.encode_text(text_tokens)
        clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
    
    CLIP_AVAILABLE = True
    print("✅ CLIP model loaded (ViT-B-32) - zero-shot image classification")
except Exception as e:
    print(f"⚠ Could not load CLIP model: {e}")
    print("⚠ Image-based department detection will not be available")

def clip_classify_image(pil_image):
    """Classify an image using CLIP zero-shot classification."""
    if not CLIP_AVAILABLE:
        return None, 0.0, {}
    
    try:
        img_tensor = clip_preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = clip_model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ clip_text_features.T).softmax(dim=-1)
        
        probs = similarity[0].numpy()
        pred_idx = int(probs.argmax())
        all_scores = {CLIP_DEPARTMENTS[i]: float(probs[i]) for i in range(3)}
        return CLIP_DEPARTMENTS[pred_idx], float(probs[pred_idx]), all_scores
    except Exception as e:
        print(f"⚠ CLIP classification error: {e}")
        return None, 0.0, {}

# =====================================================
# HAVERSINE DISTANCE FUNCTION
# =====================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2

    c = 2 * math.asin(math.sqrt(a))

    return R * c * 1000  # meters

# =====================================================
# REQUEST SCHEMA
# =====================================================
class LocationVerifyRequest(BaseModel):
    complaint_lat: float
    complaint_lon: float
    resolved_lat: float
    resolved_lon: float

class PredictComplaintRequest(BaseModel):
    description: str
    image: str = ""  # Base64 encoded image (optional)
    image_data: str = ""  # Alternative field name for base64 image

class TextAnalyzeRequest(BaseModel):
    description: str

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def tokenize_text(text):
    """Convert text to padded sequence using loaded word_index"""
    text = text.lower()
    words = text.split()
    sequence = [word_index.get(word, word_index.get("<OOV>", 1)) for word in words]
    padded = pad_sequences([sequence], maxlen=MAX_LEN, padding="post", truncating="post")
    return padded

def process_image(base64_string):
    """Decode base64 image and preprocess for model"""
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"⚠ Image processing error: {e}")
        return None

# =====================================================
# LOCATION VERIFICATION ENDPOINT
# =====================================================
@app.post("/verify-location")
def verify_location(data: LocationVerifyRequest):

    distance = haversine(
        data.complaint_lat,
        data.complaint_lon,
        data.resolved_lat,
        data.resolved_lon
    )

    # The model was trained with StandardScaler (mean=2957, std=3010)
    # We need to scale the input the same way
    TRAIN_MEAN = 2957.0
    TRAIN_STD = 3010.0
    scaled_distance = (distance - TRAIN_MEAN) / TRAIN_STD
    distance_input = np.array([[scaled_distance]])

    probability = location_verification_model.predict(
        distance_input,
        verbose=0
    )[0][0]

    return {
        "distance_meters": round(distance, 2),
        "resolved_probability": round(float(probability), 3),
        "resolved": bool(probability >= 0.5)
    }

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/")
def health():
    return {
        "status": "RuralConnect AI API running",
        "models": {
            "location_verification": "loaded",
            "multimodal_prediction": "loaded" if MULTIMODAL_AVAILABLE else "fallback (keyword-based)",
            "gpt2_text_analyzer": "loaded" if GPT2_AVAILABLE else "not available"
        },
        "departments": dept_classes,
        "severities": sev_classes
    }

# =====================================================
# GPT-2 TEXT ANALYSIS ENDPOINT
# =====================================================
@app.post("/analyze-text")
def analyze_text(data: TextAnalyzeRequest):
    """
    GPT-2 Text Analyzer:
      1. Checks if text is a negative complaint (rejects non-complaints)
      2. Predicts department from text
    """
    if not GPT2_AVAILABLE:
        # Fallback to keyword-based
        dept, sev, scores = keyword_based_prediction(data.description)
        return {
            "is_valid_complaint": True,
            "sentiment": {"is_complaint": True, "confidence": 0.7, "label": "complaint"},
            "department": {"department": dept, "confidence": 0.7, "all_scores": scores},
            "method": "keyword-fallback",
            "gpt2_available": False
        }
    
    result = gpt2_analyzer.analyze_text(data.description)
    result["method"] = "gpt2"
    return result

# =====================================================
# FUZZY KEYWORD MATCHING (typo tolerance)
# =====================================================
def fuzzy_match(word, keyword, max_distance=2):
    """Check if word is a close match to keyword (handles typos like patholes→potholes)"""
    if keyword in word or word in keyword:
        return True
    if abs(len(word) - len(keyword)) > max_distance:
        return False
    # Simple edit distance check for short words
    if len(word) < 3 or len(keyword) < 3:
        return False
    # Check if most characters overlap (allows 1-2 char typos)
    common = sum(1 for a, b in zip(word, keyword) if a == b)
    return common >= max(len(word), len(keyword)) - max_distance

def text_contains_keyword(text_lower, keyword):
    """Check if text contains a keyword, with typo tolerance for single words"""
    if keyword in text_lower:
        return True
    # For single-word keywords, try fuzzy matching against each word in text
    if " " not in keyword:
        for word in text_lower.split():
            if fuzzy_match(word, keyword):
                return True
    return False

# =====================================================
# KEYWORD-BASED FALLBACK PREDICTION
# =====================================================
def keyword_based_prediction(text):
    """Keyword-based department and severity prediction with typo tolerance"""
    text_lower = text.lower()
    
    # Department keywords
    electricity_keywords = ["electricity", "power", "light", "wire", "transformer", "voltage", "electric", "pole", "cable", "current", "shock", "sparking", "outage", "blackout"]
    water_keywords = ["water", "pipe", "tap", "supply", "leak", "drainage", "sewage", "tank", "pipeline", "flood", "flooding", "burst", "overflow", "contaminated", "drinking"]
    road_keywords = ["road", "pothole", "potholes", "patholes", "street", "highway", "pavement", "asphalt", "crack", "pit", "surface", "traffic", "bridge", "footpath", "crossing"]
    
    # Count keyword matches with fuzzy matching
    electricity_score = sum(1 for kw in electricity_keywords if text_contains_keyword(text_lower, kw))
    water_score = sum(1 for kw in water_keywords if text_contains_keyword(text_lower, kw))
    road_score = sum(1 for kw in road_keywords if text_contains_keyword(text_lower, kw))
    
    scores = {"Electricity": electricity_score, "Water": water_score, "Road": road_score}
    predicted_dept = max(scores, key=scores.get) if max(scores.values()) > 0 else "Road"
    
    # ===== SEVERITY PREDICTION =====
    predicted_sev = _predict_severity_from_text(text_lower, predicted_dept)
    
    return predicted_dept, predicted_sev, scores


def _predict_severity_from_text(text_lower, department=None):
    """
    Predict severity from text using:
      1. Explicit severity keywords (critical/high/medium)
      2. Issue-type inherent severity (potholes = High, electrocution = Critical)
      3. Quantity/intensity multipliers (many, multiple, several → boost severity)
      4. Duration indicators (since 3 days, for weeks → boost severity)
    """
    
    # --- Tier 1: Explicit critical keywords ---
    critical_keywords = [
        "urgent", "emergency", "dangerous", "hazard", "death", "dead",
        "accident", "accidents", "critical", "life", "threatening",
        "electrocution", "collapsed", "severe", "immediately", "fatal",
        "destroy", "destroyed", "casualties", "injured", "injury",
        "risk", "catch fire", "caught fire", "burning", "exploded",
        "explosion", "sinkhole", "cave in", "washed away"
    ]
    
    # --- Tier 2: High severity keywords ---
    high_keywords = [
        "immediate", "quickly", "serious", "major", "broken", "burst",
        "flooding", "flooded", "fallen", "hanging", "blocked", "overflow",
        "overflowing", "fire", "heavy", "huge", "big", "large", "massive",
        "significant", "terrible", "worst", "bad condition", "very bad",
        "leakage", "wastage", "contamination", "pollut", "stagnant",
        "no supply", "no water", "no electricity", "no power", "no light",
        "completely", "entirely", "total", "completely damaged"
    ]
    
    # --- Tier 3: Medium severity keywords ---
    medium_keywords = [
        "repair", "fix", "maintenance", "damaged", "leaking", "cracked",
        "worn", "needs attention", "issue", "problem", "complaint",
        "irregular", "disrupted", "slow", "poor", "dirty", "muddy",
        "uneven", "bumpy", "rough"
    ]
    
    # --- Issue-type inherent severity (these issues are inherently dangerous) ---
    inherently_critical = [
        "electrocution", "short circuit", "live wire", "exposed wire",
        "hanging wire", "collapsed bridge", "building collapse",
        "gas leak", "explosion", "caught fire"
    ]
    inherently_high = [
        "pothole", "potholes", "patholes", "sinkhole",
        "open manhole", "uncovered drain", "sewage overflow",
        "transformer", "sparking", "voltage fluctuation",
        "pipe burst", "water contamination", "flooding",
        "road damage", "road cave", "bridge crack",
        "fallen pole", "fallen tree", "wire hanging",
        "no water supply", "no electricity"
    ]
    
    # --- Quantity/intensity multipliers ---
    quantity_words = [
        "many", "multiple", "several", "numerous", "lots", "lot of",
        "everywhere", "all over", "entire", "whole", "every",
        "too many", "so many", "full of", "covered with",
        "spreading", "getting worse", "increasing"
    ]
    
    # --- Duration/urgency indicators ---
    duration_words = [
        "since", "for days", "for weeks", "for months", "long time",
        "still not", "yet to be", "no action", "pending",
        "repeatedly", "again and again", "every day", "daily",
        "continuous", "constantly", "persistent"
    ]
    
    # Score calculation
    base_severity = "Low"
    severity_score = 0  # 0=Low, 1=Medium, 2=High, 3=Critical
    
    # Check inherent severity of issue type (with fuzzy matching)
    for kw in inherently_critical:
        if text_contains_keyword(text_lower, kw):
            severity_score = max(severity_score, 3)
            break
    for kw in inherently_high:
        if text_contains_keyword(text_lower, kw):
            severity_score = max(severity_score, 2)
            break
    
    # Check explicit severity keywords
    if any(text_contains_keyword(text_lower, kw) for kw in critical_keywords):
        severity_score = max(severity_score, 3)
    elif any(text_contains_keyword(text_lower, kw) for kw in high_keywords):
        severity_score = max(severity_score, 2)
    elif any(text_contains_keyword(text_lower, kw) for kw in medium_keywords):
        severity_score = max(severity_score, 1)
    
    # Boost severity if quantity/intensity words are present
    has_quantity = any(text_contains_keyword(text_lower, kw) for kw in quantity_words)
    has_duration = any(text_contains_keyword(text_lower, kw) for kw in duration_words)
    
    if has_quantity:
        severity_score = min(severity_score + 1, 3)  # Boost by 1 level
        print(f"  [Severity] Quantity words detected → boosting severity")
    if has_duration:
        severity_score = min(severity_score + 1, 3)  # Boost by 1 level
        print(f"  [Severity] Duration words detected → boosting severity")
    
    severity_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    predicted_sev = severity_map[severity_score]
    
    print(f"  [Severity] Text='{text_lower[:60]}...' → {predicted_sev} (score={severity_score}, qty={has_quantity}, dur={has_duration})")
    
    return predicted_sev


# =====================================================
# CLIP IMAGE-BASED SEVERITY ANALYSIS
# =====================================================
CLIP_SEVERITY_AVAILABLE = False
clip_severity_features = None

SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]
SEVERITY_DESCRIPTIONS = [
    "a photo showing minor wear with no visible damage or safety concern",
    "a photo showing moderate damage or deterioration needing repair",
    "a photo showing severe damage with large potholes, major cracks, heavy flooding, or broken infrastructure",
    "a photo showing critical dangerous conditions, collapsed structures, exposed wires, or life-threatening hazards"
]

try:
    if CLIP_AVAILABLE:
        clip_sev_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        sev_tokens = clip_sev_tokenizer(SEVERITY_DESCRIPTIONS)
        with torch.no_grad():
            clip_severity_features = clip_model.encode_text(sev_tokens)
            clip_severity_features /= clip_severity_features.norm(dim=-1, keepdim=True)
        CLIP_SEVERITY_AVAILABLE = True
        print("\u2705 CLIP severity classifier ready (zero-shot)")
except Exception as e:
    print(f"\u26a0 Could not set up CLIP severity: {e}")


def clip_classify_severity(pil_image):
    """Classify severity of damage in image using CLIP zero-shot."""
    if not CLIP_SEVERITY_AVAILABLE:
        return None, 0.0, {}
    try:
        img_tensor = clip_preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = clip_model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ clip_severity_features.T).softmax(dim=-1)
        probs = similarity[0].numpy()
        pred_idx = int(probs.argmax())
        all_scores = {SEVERITY_LABELS[i]: float(probs[i]) for i in range(4)}
        return SEVERITY_LABELS[pred_idx], float(probs[pred_idx]), all_scores
    except Exception as e:
        print(f"\u26a0 CLIP severity error: {e}")
        return None, 0.0, {}


def combine_severity(text_severity, image_severity, image_sev_conf=0.0):
    """
    Combine text-based and image-based severity predictions.
    Image severity is weighted higher since visual damage is more objective.
    """
    sev_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    sev_from_idx = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    
    text_idx = sev_order.get(text_severity, 0)
    image_idx = sev_order.get(image_severity, 0) if image_severity else text_idx
    
    if image_severity and image_sev_conf > 0.3:
        # Image weight: 60%, Text weight: 40%
        combined_idx = round(image_idx * 0.6 + text_idx * 0.4)
        # Always take the HIGHER severity between text and image
        combined_idx = max(combined_idx, text_idx, image_idx)
    else:
        combined_idx = text_idx
    
    combined_idx = min(combined_idx, 3)
    return sev_from_idx[combined_idx]

# =====================================================
# PREDICT COMPLAINT ENDPOINT (CLIP + GPT-2)
# =====================================================
@app.post("/predict-complaint")
def predict_complaint(data: PredictComplaintRequest):
    """
    Predict department and severity from complaint text and image.
    
    Pipeline:
      1. GPT-2 checks if text is a negative complaint (rejects non-complaints)
      2. GPT-2 predicts department from text
      3. CLIP predicts department from image (zero-shot, highly accurate)
      4. Department = IMAGE prediction (CLIP)
      5. If text department != image department → NOT VALID
    """
    try:
        # ==========================================
        # STEP 0: Description Quality Check
        # ==========================================
        description_text = data.description.strip()
        desc_words = description_text.split()
        
        # Reject too-short descriptions (single words or just department names)
        vague_words = {"road", "roads", "water", "electricity", "electric", "power",
                       "light", "pipe", "wire", "pole", "supply", "drainage",
                       "pothole", "complaint", "problem", "issue", "help",
                       "bad", "broken", "damage", "fix", "repair"}
        
        # Filter out vague/generic words to see if there's real content
        meaningful_words = [w for w in desc_words if w.lower() not in vague_words and len(w) > 1]
        
        if len(desc_words) < 3 or (len(desc_words) < 5 and len(meaningful_words) < 1):
            return {
                "predicted_department": None,
                "predicted_severity": None,
                "rejected": True,
                "rejection_reason": (
                    "Your description is too short or vague. "
                    "Please describe the actual problem in detail — "
                    "for example: 'There are large potholes on MG Road near the school' "
                    "instead of just 'road' or 'pothole'."
                ),
                "sentiment": {"is_complaint": False, "label": "too_vague", "confidence": 0.99},
                "method": "description-quality-filter",
                "has_image": bool(data.image or data.image_data),
                "analysis": {
                    "department_analysis": "Rejected - description too vague",
                    "severity_analysis": "N/A",
                    "validity_analysis": "Please provide a detailed description of the issue"
                }
            }
        
        # Reject "no/not + noun" patterns that don't describe an actual problem
        # e.g. "no water pipeline", "no electricity pole", "not road"
        # These just state something doesn't exist — not a real complaint description
        desc_lower = description_text.lower()
        infrastructure_nouns = {
            "road", "roads", "water", "electricity", "electric", "power",
            "light", "lights", "pipe", "pipeline", "pipelines", "wire", "wires",
            "pole", "poles", "supply", "drainage", "transformer", "cable",
            "bridge", "tap", "tank", "meter", "line", "connection",
            "manhole", "drain", "sewer", "bulb", "lamp", "post"
        }
        
        # Check if description is just "no/not [infrastructure nouns]"
        # Pattern: starts with no/not, and remaining words are all infrastructure/vague nouns
        if desc_lower.startswith(("no ", "not ", "no. ", "noo ")):
            remaining = desc_lower.split(None, 1)[1] if len(desc_words) > 1 else ""
            remaining_words = remaining.split()
            all_nouns = all(w in infrastructure_nouns or w in vague_words for w in remaining_words)
            
            if all_nouns and len(remaining_words) <= 3:
                # This is a "no X" pattern without describing what's actually wrong
                # Check if there's a problem verb/adjective (leaking, broken, fell, burst etc.)
                problem_words = {
                    "leaking", "leaked", "burst", "broken", "fell", "fallen", "collapsed",
                    "damaged", "cracked", "flooding", "overflowing", "hanging", "sparking",
                    "burning", "blocked", "clogged", "contaminated", "disrupted",
                    "malfunctioning", "failing", "exploded", "cut", "down"
                }
                has_problem_word = any(w in problem_words for w in remaining_words)
                
                if not has_problem_word:
                    print(f"  [Quality Filter] Rejected 'no + noun' pattern: '{description_text}'")
                    return {
                        "predicted_department": None,
                        "predicted_severity": None,
                        "rejected": True,
                        "rejection_reason": (
                            "Your description only states that something doesn't exist "
                            f"('{description_text}'). Please describe the actual problem — "
                            "for example: 'Water pipeline is leaking near the market' or "
                            "'Electric pole fell down on the main road'."
                        ),
                        "sentiment": {"is_complaint": False, "label": "no_problem_described", "confidence": 0.99},
                        "method": "description-quality-filter",
                        "has_image": bool(data.image or data.image_data),
                        "analysis": {
                            "department_analysis": "Rejected - no actual problem described",
                            "severity_analysis": "N/A",
                            "validity_analysis": "Describe what is wrong, not just what exists or doesn't exist"
                        }
                    }
        
        # ==========================================
        # STEP 1: GPT-2 Sentiment Filter
        # ==========================================
        gpt2_text_result = None
        gpt2_dept = None
        gpt2_dept_conf = 0.0
        
        if GPT2_AVAILABLE:
            gpt2_text_result = gpt2_analyzer.analyze_text(data.description)
            is_complaint = gpt2_text_result.get("is_valid_complaint", True)
            
            if not is_complaint:
                return {
                    "predicted_department": None,
                    "predicted_severity": None,
                    "rejected": True,
                    "rejection_reason": gpt2_text_result.get("rejection_reason",
                        "Your description does not appear to be a complaint. "
                        "Please describe the problem or issue you are facing."),
                    "sentiment": gpt2_text_result.get("sentiment"),
                    "method": "gpt2-filtered",
                    "has_image": bool(data.image or data.image_data),
                    "analysis": {
                        "department_analysis": "Rejected - not a valid complaint",
                        "severity_analysis": "N/A",
                        "validity_analysis": "Text sentiment is not negative/complaint"
                    }
                }
            
            if gpt2_text_result.get("department"):
                gpt2_dept = gpt2_text_result["department"]["department"]
                gpt2_dept_conf = gpt2_text_result["department"]["confidence"]
        
        # ==========================================
        # STEP 2: Process image with CLIP
        # ==========================================
        image_base64 = data.image or data.image_data
        pil_image = None
        clip_dept = None
        clip_conf = 0.0
        clip_scores = {}
        
        if image_base64:
            try:
                b64_str = image_base64
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                image_bytes = base64.b64decode(b64_str)
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                print(f"⚠ Image decode error: {e}")
        
        if pil_image and CLIP_AVAILABLE:
            clip_dept, clip_conf, clip_scores = clip_classify_image(pil_image)
            print(f"  CLIP Image → {clip_dept} ({clip_conf:.0%}) | {clip_scores}")
        
        # ==========================================
        # STEP 3: No image provided → use GPT-2 text only
        # ==========================================
        if pil_image is None or not CLIP_AVAILABLE:
            dept, sev, scores = keyword_based_prediction(data.description)
            final_dept = gpt2_dept if gpt2_dept else dept
            final_conf = gpt2_dept_conf if gpt2_dept else 0.7
            
            return {
                "predicted_department": final_dept,
                "predicted_severity": sev,
                "department_confidence": round(final_conf, 3),
                "severity_confidence": 0.7,
                "overall_confidence": round(final_conf, 3),
                "rejected": False,
                "sentiment": gpt2_text_result.get("sentiment") if gpt2_text_result else None,
                "validity": {
                    "is_valid": None,
                    "score": None,
                    "image_matches_description": None,
                    "message": "Cannot validate - no image provided"
                },
                "method": "gpt2 (no image)" if gpt2_dept else "keyword-based (no image)",
                "has_image": False,
                "analysis": {
                    "department_analysis": f"Text analysis: {final_dept}",
                    "severity_analysis": f"Severity: {sev}",
                    "validity_analysis": "Upload an image for cross-validation"
                }
            }
        
        # ==========================================
        # STEP 4: DEPARTMENT = IMAGE (CLIP), TEXT = GPT-2
        # ==========================================
        text_dept = gpt2_dept if gpt2_dept else keyword_based_prediction(data.description)[0]
        text_dept_conf = gpt2_dept_conf if gpt2_dept else 0.6
        
        # Department is ALWAYS from the image
        final_dept = clip_dept
        dept_confidence = clip_conf
        
        # Check if text matches image
        text_image_match = (text_dept == clip_dept)
        
        print(f"  Text dept: {text_dept} ({text_dept_conf:.0%}), Image dept: {clip_dept} ({clip_conf:.0%}), Match: {text_image_match}")
        
        # ==========================================
        # STEP 5: Severity from text + image
        # ==========================================
        _, keyword_sev, keyword_scores = keyword_based_prediction(data.description)
        
        # Also get severity from CLIP image analysis
        image_sev = None
        image_sev_conf = 0.0
        image_sev_scores = {}
        if pil_image and CLIP_SEVERITY_AVAILABLE:
            image_sev, image_sev_conf, image_sev_scores = clip_classify_severity(pil_image)
            print(f"  CLIP Severity → {image_sev} ({image_sev_conf:.0%}) | {image_sev_scores}")
        
        # Combine text + image severity (takes the higher of the two)
        final_sev = combine_severity(keyword_sev, image_sev, image_sev_conf)
        sev_confidence = max(0.75, image_sev_conf) if image_sev else 0.75
        print(f"  Final Severity: text={keyword_sev}, image={image_sev}, combined={final_sev}")
        
        # ==========================================
        # STEP 6: VALIDITY — text must match image
        # ==========================================
        if text_image_match:
            is_valid = True
            validity_score = max(clip_conf, 0.92)
            validity_message = f"Valid: Description matches image ({clip_dept} department)"
        else:
            is_valid = False
            validity_score = round(1.0 - clip_conf, 3)
            validity_message = (
                f"Not Valid: Your description suggests '{text_dept}' department, "
                f"but the uploaded image shows a '{clip_dept}' issue. "
                f"Please upload a matching image or correct your description."
            )
            print(f"  ⚠ NOT VALID: Text='{text_dept}' vs Image='{clip_dept}'")
        
        overall_confidence = (dept_confidence + sev_confidence + (validity_score if is_valid else 0.3)) / 3
        
        return {
            "predicted_department": final_dept,
            "predicted_severity": final_sev,
            "rejected": False,
            "department_confidence": round(dept_confidence, 3),
            "severity_confidence": round(sev_confidence, 3),
            "overall_confidence": round(overall_confidence, 3),
            "sentiment": gpt2_text_result.get("sentiment") if gpt2_text_result else None,
            "cross_validation": {
                "status": "confirmed" if text_image_match else "mismatch",
                "text_department": text_dept,
                "text_confidence": round(text_dept_conf, 3),
                "image_department": clip_dept,
                "image_confidence": round(clip_conf, 3),
                "agreement": text_image_match,
                "message": (
                    f"Text and image both indicate: {final_dept}"
                    if text_image_match
                    else f"MISMATCH: Text says '{text_dept}' but image shows '{clip_dept}'"
                )
            },
            "validity": {
                "is_valid": is_valid,
                "score": round(validity_score, 3),
                "image_matches_description": text_image_match,
                "message": validity_message
            },
            "method": "clip + gpt2",
            "has_image": True,
            "image_analysis": {
                "predicted_department": clip_dept,
                "confidence": round(clip_conf, 3),
                "all_scores": {k: round(v, 3) for k, v in clip_scores.items()},
                "model": "CLIP ViT-B-32"
            },
            "text_analysis": {
                "method": "gpt2" if gpt2_dept else "keywords",
                "predicted_department": text_dept,
                "department_confidence": round(text_dept_conf, 3),
                "predicted_severity": keyword_sev,
                "keyword_scores": keyword_scores
            },
            "analysis": {
                "department_analysis": f"Image shows: {clip_dept} ({clip_conf:.0%} confidence)",
                "severity_analysis": f"Severity: {final_sev}",
                "validity_analysis": f"{'Valid' if is_valid else 'Not Valid'}: Text='{text_dept}' vs Image='{clip_dept}'"
            }
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        try:
            dept, sev, scores = keyword_based_prediction(data.description)
            return {
                "predicted_department": dept,
                "predicted_severity": sev,
                "department_confidence": 0.5,
                "severity_confidence": 0.5,
                "overall_confidence": 0.5,
                "validity": {
                    "is_valid": None,
                    "score": None,
                    "message": "Model error - used keyword fallback"
                },
                "method": "keyword-based (fallback)",
                "has_image": False,
                "error_details": str(e)
            }
        except:
            return {
                "error": str(e),
                "predicted_department": None,
                "predicted_severity": None
            }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
