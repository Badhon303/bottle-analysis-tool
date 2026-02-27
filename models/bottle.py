# ============================================================================
# models/bottle.py - Bottle Database Models
# ============================================================================

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class BottleLabel(Base):
    """Known bottle labels (e.g., 'Mojo 250mL', 'Pran 500mL')"""
    __tablename__ = "bottle_labels"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # e.g., "Mojo 250mL"
    brand = Column(String, nullable=True)            # e.g., "Mojo"
    volume_ml = Column(Integer, nullable=True)       # e.g., 250
    material = Column(String, default="PET")         # e.g., "PET", "HDPE"
    empty_weight_grams = Column(Float, nullable=True)  # Known weight when empty
    description = Column(String, nullable=True)
    
    # Reference embedding (average of all confirmed bottles of this type)
    reference_embedding = Column(JSON, nullable=True)
    sample_count = Column(Integer, default=0)  # How many samples used for embedding
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bottles = relationship("DetectedBottle", back_populates="label")


class DetectedBottle(Base):
    """Individual bottles detected in videos"""
    __tablename__ = "detected_bottles"
    
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), index=True)
    
    # Tracking info
    track_id = Column(Integer)  # ID from tracker within the video
    
    # Image data
    image_path = Column(String)  # Path to cropped bottle image
    best_frame_number = Column(Integer)
    sharpness_score = Column(Float)
    
    # Embedding for matching
    embedding = Column(JSON)
    
    # Classification
    label_id = Column(Integer, ForeignKey("bottle_labels.id"), nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, default="pending")  # pending, auto_labeled, confirmed, new_label
    
    # Cluster (for grouping unknowns)
    cluster_id = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    batch = relationship("Batch", back_populates="bottles")
    label = relationship("BottleLabel", back_populates="bottles")


# Note: Table creation is now handled in main.py after all models are imported