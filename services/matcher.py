# ============================================================================
# services/matcher.py - Match Bottles to Known Labels
# ============================================================================

from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Tuple, List, Dict
from sqlalchemy.orm import Session
from dataclasses import dataclass
import numpy as np

@dataclass
class MatchResult:
    label_id: Optional[int]
    label_name: Optional[str]
    confidence: float
    status: str  # "high_confidence", "medium_confidence", "no_match"
    empty_weight_grams: Optional[float] = None  # Weight of the bottle label

class BottleMatcher:
    """Match bottle embeddings to known labels"""
    
    def __init__(
        self,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.60
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.label_embeddings = {}  # label_id -> embedding
        self.label_names = {}  # label_id -> name
        self.label_weights = {}  # label_id -> empty_weight_grams
    
    def load_labels(self, db: Session):
        """Load all known label embeddings from database"""
        from models.bottle import BottleLabel  # Import here to avoid circular
        
        labels = db.query(BottleLabel).filter(
            BottleLabel.reference_embedding.isnot(None)
        ).all()
        
        for label in labels:
            self.label_embeddings[label.id] = np.array(label.reference_embedding)
            self.label_names[label.id] = label.name
            self.label_weights[label.id] = label.empty_weight_grams
    
    def match(self, embedding: np.ndarray) -> MatchResult:
        """Find best matching label for an embedding"""
        if not self.label_embeddings:
            return MatchResult(
                label_id=None,
                label_name=None,
                confidence=0.0,
                status="no_match",
                empty_weight_grams=None
            )
        
        best_label_id = None
        best_similarity = -1
        
        for label_id, ref_embedding in self.label_embeddings.items():
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                ref_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_label_id = label_id
        
        # Determine confidence level
        if best_similarity >= self.high_threshold:
            status = "high_confidence"
        elif best_similarity >= self.medium_threshold:
            status = "medium_confidence"
        else:
            status = "no_match"
            best_label_id = None
        
        return MatchResult(
            label_id=best_label_id,
            label_name=self.label_names.get(best_label_id),
            confidence=float(best_similarity),
            status=status,
            empty_weight_grams=self.label_weights.get(best_label_id)
        )
    
    def get_all_similarities(self, embedding: np.ndarray) -> List[Dict]:
        """
        Get similarity scores between an embedding and all known labels.
        Returns list of dicts sorted by similarity (highest first).
        """
        if not self.label_embeddings:
            return []
        
        similarities = []
        for label_id, ref_embedding in self.label_embeddings.items():
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                ref_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append({
                "label_id": label_id,
                "label_name": self.label_names.get(label_id),
                "similarity": float(sim)
            })
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities
    
    def compute_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of embeddings.
        Returns NxN matrix where entry [i,j] is similarity between embeddings i and j.
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Stack embeddings into matrix
        embedding_matrix = np.vstack([e.reshape(1, -1) for e in embeddings])
        
        # Compute pairwise cosine similarities
        return cosine_similarity(embedding_matrix)
    
    def update_label_embedding(self, db: Session, label_id: int, new_embedding: np.ndarray):
        """Update a label's reference embedding with new sample (running average)"""
        from models.bottle import BottleLabel
        
        label = db.query(BottleLabel).filter(BottleLabel.id == label_id).first()
        if not label:
            return
        
        if label.reference_embedding is None:
            # First sample
            label.reference_embedding = new_embedding.tolist()
            label.sample_count = 1
        else:
            # Running average
            old_embedding = np.array(label.reference_embedding)
            n = label.sample_count
            updated = (old_embedding * n + new_embedding) / (n + 1)
            label.reference_embedding = updated.tolist()
            label.sample_count = n + 1
        
        db.commit()
        
        # Update local cache
        self.label_embeddings[label_id] = np.array(label.reference_embedding)


