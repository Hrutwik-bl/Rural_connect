"""
GPT-2 Text Analyzer - Inference Module
=======================================
Loads fine-tuned GPT-2 models for:
  1. Sentiment detection (complaint vs non-complaint)
  2. Department classification (Electricity / Road / Water)
  3. Cross-validation with CNN image predictions
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# =====================================================
# CONFIGURATION
# =====================================================
MAX_LEN = 64  # Must match training MAX_LEN
MODELS_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Department label mapping
DEPT_LABELS = {0: "Electricity", 1: "Road", 2: "Water"}
SENTIMENT_LABELS = {0: "not_complaint", 1: "complaint"}


def _fuzzy_match(word, keyword, max_distance=2):
    """Check if word is a close match to keyword (handles typos like patholes→potholes)"""
    if keyword in word or word in keyword:
        return True
    if abs(len(word) - len(keyword)) > max_distance:
        return False
    if len(word) < 3 or len(keyword) < 3:
        return False
    common = sum(1 for a, b in zip(word, keyword) if a == b)
    return common >= max(len(word), len(keyword)) - max_distance

def _text_contains_keyword(text_lower, keyword):
    """Check if text contains a keyword, with typo tolerance for single words"""
    if keyword in text_lower:
        return True
    if " " not in keyword:
        for word in text_lower.split():
            if _fuzzy_match(word, keyword):
                return True
    return False


class GPT2TextAnalyzer:
    """
    GPT-2 based text analyzer for RuralConnect.
    """
    
    def __init__(self):
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.dept_model = None
        self.dept_tokenizer = None
        self.is_loaded = False
        
    def _setup_tokenizer(self, tokenizer):
        """Ensure tokenizer uses eos_token as pad with left padding (matches training)"""
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer
        
    def load_models(self):
        """Load both GPT-2 models"""
        try:
            # Load sentiment model
            sentiment_dir = MODELS_DIR / "gpt2_sentiment"
            if sentiment_dir.exists():
                self.sentiment_tokenizer = GPT2Tokenizer.from_pretrained(str(sentiment_dir))
                self._setup_tokenizer(self.sentiment_tokenizer)
                self.sentiment_model = GPT2ForSequenceClassification.from_pretrained(str(sentiment_dir))
                self.sentiment_model.config.pad_token_id = self.sentiment_tokenizer.pad_token_id
                self.sentiment_model.to(DEVICE)
                self.sentiment_model.eval()
                print(f"✅ GPT-2 Sentiment model loaded")
            else:
                print(f"⚠ Sentiment model not found at {sentiment_dir}")
                return False
            
            # Load department model
            dept_dir = MODELS_DIR / "gpt2_department"
            if dept_dir.exists():
                self.dept_tokenizer = GPT2Tokenizer.from_pretrained(str(dept_dir))
                self._setup_tokenizer(self.dept_tokenizer)
                self.dept_model = GPT2ForSequenceClassification.from_pretrained(str(dept_dir))
                self.dept_model.config.pad_token_id = self.dept_tokenizer.pad_token_id
                self.dept_model.to(DEVICE)
                self.dept_model.eval()
                print(f"✅ GPT-2 Department model loaded")
            else:
                print(f"⚠ Department model not found at {dept_dir}")
                return False
            
            self.is_loaded = True
            print("✅ GPT-2 Text Analyzer ready")
            return True
            
        except Exception as e:
            print(f"❌ Error loading GPT-2 models: {e}")
            import traceback; traceback.print_exc()
            self.is_loaded = False
            return False
    
    def _detect_negation(self, text: str) -> dict:
        """
        Rule-based negation detection to catch phrases like:
          'no water leakage', 'not any pothole', 'road is not damaged'
        These negate the complaint meaning and should be rejected.
        
        Returns None if no negation detected, otherwise a result dict.
        """
        text_lower = text.lower().strip()
        
        # Complaint keywords that could be negated
        complaint_keywords = [
            "water leakage", "leakage", "leaking", "leak",
            "pothole", "potholes", "road damage", "damaged", "broken",
            "electricity problem", "power outage", "power cut", "power failure",
            "flooding", "overflow", "pipe burst", "drainage", "sewage",
            "crack", "cracks", "issue", "problem", "complaint",
            "shortage", "contamination", "disruption", "accident",
            "wire hanging", "pole damage", "transformer",
            "water supply problem", "electricity issue", "road problem",
            "water problem", "blocked", "dirty water",
            "water logging", "street light issue"
        ]
        
        # Strong negation starters - "no X", "not any X"
        negation_starters = [
            "no ", "not ", "not any ", "there is no ", "there are no ",
            "no issues", "no problem", "nothing wrong", "nothing is broken",
            "nothing to complain", "no need to complain", "all fine",
            "everything is fine", "everything works", "everything is normal",
            "no complaints", "all is good", "no damage",
            "not facing any", "no issues found", "no problems reported",
        ]
        
        # Check: text starts with negation + contains complaint keyword
        for neg in negation_starters:
            if text_lower.startswith(neg):
                for kw in complaint_keywords:
                    if kw in text_lower:
                        return {
                            "is_complaint": False,
                            "confidence": 0.95,
                            "label": "not_complaint",
                            "negation_detected": True,
                            "pattern": f"'{neg.strip()}' + '{kw}'",
                            "scores": {"not_complaint": 0.95, "complaint": 0.05}
                        }
        
        # Check: "X is not Y" pattern (subject + is/are + not + complaint_word)
        negation_verb_patterns = [
            "is not ", "is not being ", "are not ", "is fine",
            "is okay", "is working", "is stable", "is normal",
            "is good", "is perfect", "is smooth",
            "works fine", "working fine", "working properly",
            "running fine", "running smoothly",
            "in good condition", "is working properly",
        ]
        for pattern in negation_verb_patterns:
            if pattern in text_lower:
                for kw in complaint_keywords:
                    if kw in text_lower:
                        return {
                            "is_complaint": False,
                            "confidence": 0.92,
                            "label": "not_complaint",
                            "negation_detected": True,
                            "pattern": f"'{pattern.strip()}' + '{kw}'",
                            "scores": {"not_complaint": 0.92, "complaint": 0.08}
                        }
        
        # Purely positive short texts (< 8 words) with negation and complaint keywords
        words = text_lower.split()
        if len(words) <= 8:
            has_negation = any(w in ["no", "not", "nothing", "none", "never", "nor"] for w in words)
            has_complaint_kw = any(kw in text_lower for kw in complaint_keywords)
            if has_negation and has_complaint_kw:
                return {
                    "is_complaint": False,
                    "confidence": 0.90,
                    "label": "not_complaint",
                    "negation_detected": True,
                    "pattern": "short_text_with_negation",
                    "scores": {"not_complaint": 0.90, "complaint": 0.10}
                }
        
        return None  # No negation detected, use model
    
    def _detect_complaint_keywords(self, text: str) -> dict:
        """
        Rule-based complaint detection to catch obvious complaints that
        the GPT-2 model may incorrectly reject.
        
        If the text contains strong complaint/problem keywords and no negation,
        it should be treated as a complaint.
        
        Returns None if no strong keywords found, otherwise a result dict.
        """
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Negation words - if present alongside complaint keywords, let the model decide
        negation_words = {"no", "not", "nothing", "none", "never", "nor", "fine", "good", "okay", "perfect", "working"}
        has_negation = any(w in negation_words for w in words)
        
        # If text has negation, don't override the model
        if has_negation:
            return None
        
        # Strong complaint / problem indicators (with common typo variants)
        strong_complaint_keywords = [
            "pothole", "potholes", "patholes", "pathole",
            "broken", "damaged", "damage", "collapsed", "fallen",
            "burst", "flooding", "flooded", "overflow", "overflowing",
            "leaking", "leak", "leakage", "contaminated", "dirty",
            "accident", "accidents", "dangerous", "hazard", "hazardous",
            "electrocution", "shock", "sparking", "fire", "burning",
            "stuck", "blocked", "clogged", "sewage", "stagnant",
            "outage", "blackout", "power cut", "power failure", "no supply",
            "no water", "no electricity", "no power", "no light",
            "hanging wire", "exposed wire", "short circuit",
            "crack", "cracks", "cracked", "eroded", "erosion",
            "waterlogging", "water logging", "sinkhole", "cave in",
            "bad condition", "worst condition", "terrible", "horrible",
            "need repair", "needs repair", "fix it", "urgent", "emergency",
            "complaint", "complain", "problem", "issue", "grievance",
            "pipe", "wire", "pole",
            "supply", "drainage", "transformer", "voltage",
        ]
        
        # Contextual complaint phrases (multi-word)
        complaint_phrases = [
            "causing accidents", "causing problems", "causing damage",
            "not working", "stopped working", "failed", "malfunctioning",
            "out of order", "supply disrupted", "supply cut",
            "needs attention", "needs fixing", "needs repair",
            "very bad", "very poor", "in bad shape",
            "people are suffering", "residents are facing",
        ]
        
        # Reject very short/vague descriptions (just a word or two like "road", "water")
        if len(words) <= 2:
            return None  # Too vague, let the model decide (likely not_complaint)
        
        # Count strong keyword matches (with fuzzy matching for typo tolerance)
        keyword_matches = sum(1 for kw in strong_complaint_keywords if _text_contains_keyword(text_lower, kw))
        phrase_matches = sum(1 for ph in complaint_phrases if ph in text_lower)
        total_matches = keyword_matches + phrase_matches
        
        if total_matches >= 2:
            # Multiple complaint keywords → definitely a complaint
            confidence = min(0.95, 0.80 + total_matches * 0.03)
            return {
                "is_complaint": True,
                "confidence": round(confidence, 4),
                "label": "complaint",
                "keyword_override": True,
                "matched_keywords": total_matches,
                "scores": {"not_complaint": round(1 - confidence, 4), "complaint": round(confidence, 4)}
            }
        elif total_matches == 1:
            # Single strong keyword → likely a complaint
            confidence = 0.78
            return {
                "is_complaint": True,
                "confidence": confidence,
                "label": "complaint",
                "keyword_override": True,
                "matched_keywords": total_matches,
                "scores": {"not_complaint": round(1 - confidence, 4), "complaint": confidence}
            }
        
        return None  # No strong complaint keywords found, use model
    
    def analyze_sentiment(self, text: str) -> dict:
        """
        Classify text as complaint (negative) or not-complaint.
        Uses rule-based negation detection + GPT-2 model + keyword override.
        
        Pipeline:
          1. Check for negation patterns (reject non-complaints)
          2. Run GPT-2 sentiment model
          3. If model says not_complaint, check keyword override (catch false rejections)
        
        Returns:
            dict with keys: is_complaint, confidence, label
        """
        if not self.is_loaded or self.sentiment_model is None:
            # If model not loaded, use keyword detection as fallback
            keyword_result = self._detect_complaint_keywords(text)
            if keyword_result is not None:
                return keyword_result
            return {"is_complaint": None, "confidence": 0.0, "label": "unknown", "error": "Model not loaded"}
        
        # Step 1: Check for negation patterns first (rule-based)
        negation_result = self._detect_negation(text)
        if negation_result is not None:
            print(f"  [Negation Filter] '{text}' → NOT_COMPLAINT (pattern: {negation_result.get('pattern')})")
            return negation_result
        
        try:
            # Step 2: Run GPT-2 sentiment model
            encoding = self.sentiment_tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            with torch.no_grad():
                outputs = self.sentiment_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()
            
            is_complaint = (predicted_class == 1)
            label = SENTIMENT_LABELS[predicted_class]
            
            # Step 3: If model says NOT a complaint, check keyword override
            # This catches cases where model incorrectly rejects valid complaints
            if not is_complaint:
                keyword_result = self._detect_complaint_keywords(text)
                if keyword_result is not None:
                    print(f"  [Keyword Override] Model said not_complaint ({confidence:.2%}), "
                          f"but keywords say complaint ({keyword_result['matched_keywords']} matches). Overriding.")
                    keyword_result["model_overridden"] = True
                    keyword_result["original_model_prediction"] = {
                        "label": label,
                        "confidence": round(confidence, 4),
                        "scores": {
                            "not_complaint": round(probs[0].item(), 4),
                            "complaint": round(probs[1].item(), 4)
                        }
                    }
                    return keyword_result
            
            return {
                "is_complaint": is_complaint,
                "confidence": round(confidence, 4),
                "label": label,
                "scores": {
                    "not_complaint": round(probs[0].item(), 4),
                    "complaint": round(probs[1].item(), 4)
                }
            }
            
        except Exception as e:
            # On error, try keyword detection as fallback
            keyword_result = self._detect_complaint_keywords(text)
            if keyword_result is not None:
                return keyword_result
            return {"is_complaint": None, "confidence": 0.0, "label": "error", "error": str(e)}
    
    def _keyword_department_check(self, text: str) -> dict:
        """
        Rule-based department detection using keywords.
        Returns a department result dict if strong keywords found, else None.
        """
        text_lower = text.lower()
        
        road_keywords = ["road", "pothole", "potholes", "street", "highway", "pavement",
                         "asphalt", "crack", "pit", "surface", "traffic", "bridge",
                         "footpath", "crossing", "lane", "path", "patholes"]
        water_keywords = ["water", "pipe", "tap", "supply", "leak", "drainage",
                          "sewage", "tank", "pipeline", "flood", "flooding", "burst",
                          "overflow", "contaminated", "drinking", "borewell", "well"]
        electricity_keywords = ["electricity", "power", "light", "wire", "transformer",
                                "voltage", "electric", "pole", "cable", "current",
                                "shock", "sparking", "outage", "blackout", "bulb"]
        
        road_score = sum(1 for kw in road_keywords if kw in text_lower)
        water_score = sum(1 for kw in water_keywords if kw in text_lower)
        elec_score = sum(1 for kw in electricity_keywords if kw in text_lower)
        
        scores = {"Road": road_score, "Water": water_score, "Electricity": elec_score}
        max_score = max(scores.values())
        
        if max_score == 0:
            return None
        
        best_dept = max(scores, key=scores.get)
        return {
            "department": best_dept,
            "keyword_score": max_score,
            "all_keyword_scores": scores
        }
    
    def predict_department(self, text: str) -> dict:
        """
        Classify complaint text into department: Electricity / Road / Water.
        Uses GPT-2 model + keyword cross-validation to fix misclassifications.
        
        Should only be called AFTER confirming text is a complaint.
        
        Returns:
            dict with keys: department, confidence, all_scores
        """
        if not self.is_loaded or self.dept_model is None:
            # Fallback to keyword-based
            kw_result = self._keyword_department_check(text)
            if kw_result:
                return {
                    "department": kw_result["department"],
                    "confidence": 0.7,
                    "all_scores": {k: round(v / max(sum(kw_result["all_keyword_scores"].values()), 1), 4) for k, v in kw_result["all_keyword_scores"].items()}
                }
            return {"department": None, "confidence": 0.0, "error": "Model not loaded"}
        
        try:
            encoding = self.dept_tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            with torch.no_grad():
                outputs = self.dept_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()
            department = DEPT_LABELS[predicted_class]
            
            all_scores = {
                DEPT_LABELS[i]: round(probs[i].item(), 4)
                for i in range(len(DEPT_LABELS))
            }
            
            # Cross-validate with keywords: if model is uncertain or keywords strongly disagree, override
            kw_result = self._keyword_department_check(text)
            if kw_result and kw_result["department"] != department:
                kw_dept = kw_result["department"]
                kw_score = kw_result["keyword_score"]
                
                # Override if: keywords have strong match AND model confidence is low
                # OR keywords have very strong match (2+ keywords)
                if (kw_score >= 2) or (kw_score >= 1 and confidence < 0.65):
                    print(f"  [Dept Override] Model: {department} ({confidence:.2%}), "
                          f"Keywords: {kw_dept} ({kw_score} matches). Using keywords.")
                    department = kw_dept
                    confidence = max(confidence, 0.75 + kw_score * 0.05)
                    all_scores[kw_dept] = max(all_scores.get(kw_dept, 0), confidence)
            
            return {
                "department": department,
                "confidence": round(confidence, 4),
                "all_scores": all_scores
            }
            
        except Exception as e:
            kw_result = self._keyword_department_check(text)
            if kw_result:
                return {
                    "department": kw_result["department"],
                    "confidence": 0.7,
                    "all_scores": {k: round(v / max(sum(kw_result["all_keyword_scores"].values()), 1), 4) for k, v in kw_result["all_keyword_scores"].items()}
                }
            return {"department": None, "confidence": 0.0, "error": str(e)}
    
    def analyze_text(self, text: str) -> dict:
        """
        Full text analysis pipeline:
          1. Check if text is a negative complaint
          2. If yes, classify department
          3. Return combined results
        """
        result = {
            "text": text,
            "sentiment": None,
            "department": None,
            "is_valid_complaint": False,
            "gpt2_available": self.is_loaded
        }
        
        if not self.is_loaded:
            result["error"] = "GPT-2 models not loaded"
            return result
        
        # Step 1: Sentiment analysis
        sentiment = self.analyze_sentiment(text)
        result["sentiment"] = sentiment
        
        if not sentiment.get("is_complaint"):
            result["is_valid_complaint"] = False
            result["rejection_reason"] = (
                "Text does not appear to be a complaint/negative description. "
                "Please provide a description of the issue you are facing."
            )
            return result
        
        # Step 2: Department classification (only for complaints)
        department = self.predict_department(text)
        result["department"] = department
        result["is_valid_complaint"] = True
        
        return result
    
    def cross_validate_with_cnn(self, text: str, cnn_department: str, cnn_confidence: float) -> dict:
        """
        Cross-validate GPT-2 text department prediction with CNN image prediction.
        
        Args:
            text: complaint description
            cnn_department: department predicted by CNN from image
            cnn_confidence: CNN confidence score
            
        Returns:
            dict with cross-validation results
        """
        # Get GPT-2 predictions
        text_analysis = self.analyze_text(text)
        
        if not text_analysis["is_valid_complaint"]:
            return {
                "cross_validation": "rejected",
                "reason": text_analysis.get("rejection_reason", "Not a valid complaint"),
                "text_analysis": text_analysis,
                "cnn_department": cnn_department,
                "agreement": False,
                "final_department": None
            }
        
        gpt2_dept = text_analysis["department"]["department"]
        gpt2_conf = text_analysis["department"]["confidence"]
        
        # Check agreement
        agreement = (gpt2_dept == cnn_department)
        
        # Decision logic for final department
        if agreement:
            # Both models agree - high confidence
            final_dept = gpt2_dept
            final_confidence = max(gpt2_conf, cnn_confidence)
            match_status = "confirmed"
            message = f"Both GPT-2 text analysis and CNN image analysis agree: {final_dept}"
        else:
            # Models disagree - use weighted decision
            # GPT-2 is generally better for text, CNN for images
            # Weight: GPT-2 text = 0.55, CNN image = 0.45 (text slightly preferred for dept)
            gpt2_weight = 0.55
            cnn_weight = 0.45
            
            gpt2_weighted = gpt2_conf * gpt2_weight
            cnn_weighted = cnn_confidence * cnn_weight
            
            if gpt2_weighted >= cnn_weighted:
                final_dept = gpt2_dept
                final_confidence = gpt2_conf
                match_status = "text_preferred"
                message = (f"Disagreement: GPT-2 says {gpt2_dept} ({gpt2_conf:.1%}), "
                          f"CNN says {cnn_department} ({cnn_confidence:.1%}). "
                          f"Using text-based prediction: {final_dept}")
            else:
                final_dept = cnn_department
                final_confidence = cnn_confidence
                match_status = "image_preferred"
                message = (f"Disagreement: GPT-2 says {gpt2_dept} ({gpt2_conf:.1%}), "
                          f"CNN says {cnn_department} ({cnn_confidence:.1%}). "
                          f"Using image-based prediction: {final_dept}")
        
        return {
            "cross_validation": match_status,
            "agreement": agreement,
            "final_department": final_dept,
            "final_confidence": round(final_confidence, 4),
            "message": message,
            "text_analysis": {
                "department": gpt2_dept,
                "confidence": gpt2_conf,
                "all_scores": text_analysis["department"]["all_scores"],
                "sentiment": text_analysis["sentiment"]
            },
            "image_analysis": {
                "department": cnn_department,
                "confidence": round(cnn_confidence, 4)
            }
        }
