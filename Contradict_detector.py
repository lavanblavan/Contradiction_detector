"""
Semantic Contradiction Detector
Assignment - Part 2
"""
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
from transformers import pipeline
import torch.nn.functional as F

@dataclass
class ContradictionResult:
    has_contradiction: bool
    confidence: float
    contradicting_pairs: List[Tuple[str, str]]
    explanation: str

class SemanticContradictionDetector:
    """
    Detects semantic contradictions within a single document.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the detector with specified model.
        
        Args:
            model_name: Model identifier or path
        """
        # TODO: Initialize your model(s) here
        # Setup Device
        self.device = 0 if torch.cuda.is_available() else -1
        self.dtype = torch.float16 if self.device == 0 else torch.float32
        
        # Expert 1: Sentiment/General contradiction
        self.roberta = pipeline(
            "text-classification", 
            model="roberta-large-mnli", 
            device=self.device,
            torch_dtype=self.dtype
        )
        
        # Expert 2: Logical/Numeric/Temporal contradiction
        self.deberta = pipeline(
            "text-classification", 
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", 
            device=self.device,
            torch_dtype=self.dtype
        )
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text into analyzable units.
        
        Args:
            text: Raw review text
            
        Returns:
            List of sentences or semantic units
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 8]
        return sentences
    
    def extract_claims(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extract factual claims from sentences.
        
        Args:
            sentences: Preprocessed sentences
            
        Returns:
            List of claim dictionaries with metadata
        """
        # TODO: Implement claim extraction
        return [{"text": s, "metadata": {"length": len(s)}} for s in sentences]
    
    def check_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:
        """
        Check if two claims contradict each other.
        
        Args:
            claim_a: First claim
            claim_b: Second claim
            
        Returns:
            Tuple of (is_contradiction, confidence)
        """
        # TODO: Implement contradiction logic
        """Hybrid logic reasoning using both Experts."""
        sent_a, sent_b = claim_a["text"], claim_b["text"]
        
        # Expert 1 Prediction
        res_r = self.roberta(f"{sent_a} </s></s> {sent_b}")[0]
        score_r = res_r['score'] if res_r['label'] == 'CONTRADICTION' else 0
        
        # Expert 2 Prediction
        res_d = self.deberta(f"{sent_a} [SEP] {sent_b}")[0]
        score_d = res_d['score'] if res_d['label'] == 'contradiction' else 0

        # Ensemble: Take the highest confidence from either expert
        confidence = max(score_r, score_d)
        is_contradiction = confidence > 0.62
        
        return is_contradiction, confidence
    
    def analyze(self, text: str) -> ContradictionResult:
        """
        Main analysis pipeline.
        
        Args:
            text: Review text to analyze
            
        Returns:
            ContradictionResult with findings
        """
        # TODO: Implement full pipeline
        """Main analysis pipeline: Preprocess -> Extract -> Check Pairs."""
        sentences = self.preprocess(text)
        claims = self.extract_claims(sentences)
        
        found_pairs = []
        max_conf = 0.0
        
        # Compare all pairs
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                is_contra, conf = self.check_contradiction(claims[i], claims[j])
                
                # Track the highest confidence score found in the document
                if conf > max_conf:
                    max_conf = conf
                
                if is_contra:
                    found_pairs.append((claims[i]["text"], claims[j]["text"]))
        
        has_contra = len(found_pairs) > 0
        
        # If no contradiction is found, confidence reflects how "consistent" it is
        # If contradiction is found, confidence is the strength of the strongest clash
        final_confidence = max_conf if has_contra else (1.0 - max_conf)
        
        explanation = f"Found {len(found_pairs)} contradicting claim(s)." if has_contra else "No contradictions detected."
        
        return ContradictionResult(
            has_contradiction=has_contra,
            confidence=round(float(final_confidence), 4),
            contradicting_pairs=found_pairs,
            explanation=explanation
        )

def evaluate(detector: SemanticContradictionDetector, 
             test_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate detector performance.
    
    Args:
        detector: Initialized detector
        test_data: List of test samples with ground truth
        
    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    # TODO: Implement evaluation
    """Calculate standard evaluation metrics."""
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for item in test_data:
        result = detector.analyze(item["text"])
        pred = result.has_contradiction
        actual = item["has_contradiction"]
        
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and not actual: tn += 1
        else: fn += 1
    
    accuracy = (tp + tn) / len(test_data) if test_data else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


if __name__ == "__main__":
    # Initialize detector
    detector = SemanticContradictionDetector()
    
    # Run on sample data
    from sample_data import SAMPLE_REVIEWS
    
    for review in SAMPLE_REVIEWS:
        result = detector.analyze(review["text"])
        print(f"\n--- Review {review['id']} ---")
        print(f"Result: {'CONTRADICTION FOUND' if result.has_contradiction else 'Consistent'}")
        print(f"Confidence Score: {result.confidence * 100}%")
        if result.has_contradiction:
            for p1, p2 in result.contradicting_pairs:
                print(f"  Conflict: \"{p1}\" VS \"{p2}\"")

    # 4. Evaluate
    metrics = evaluate(detector, SAMPLE_REVIEWS)
    print(f"\nFinal Metrics: {metrics}")