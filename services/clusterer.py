# ============================================================================
# services/clusterer.py - Cluster Unknown Bottles
# ============================================================================

from sklearn.cluster import HDBSCAN
from typing import List, Dict
import numpy as np

class BottleClusterer:
    """Cluster unknown bottles by visual similarity"""
    
    def __init__(self, min_cluster_size: int = 2, min_samples: int = 2):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
    
    def cluster(self, embeddings: np.ndarray, bottle_ids: List[int]) -> Dict[int, int]:
        """
        Cluster embeddings and return mapping of bottle_id -> cluster_id
        Returns -1 for outliers (bottles that don't fit any cluster)
        """
        if len(embeddings) < self.min_cluster_size:
            return {bid: -1 for bid in bottle_ids}
        
        # Normalize embeddings for cosine distance
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # HDBSCAN clustering
        clustering = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='cosine'
        ).fit(normalized)
        
        return {
            bottle_id: int(cluster_id)
            for bottle_id, cluster_id in zip(bottle_ids, clustering.labels_)
        }