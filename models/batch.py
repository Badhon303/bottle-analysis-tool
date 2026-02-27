# ============================================================================
# models/batch.py - Batch Database Models
# ============================================================================

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Batch(Base):
    """A batch of bottles from a supplier"""
    __tablename__ = "batches"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Batch info
    name = Column(String, index=True)  # e.g., "Batch-2024-01-15-001"
    supplier = Column(String, nullable=True)
    
    # Video info
    video_path = Column(String)
    video_duration_seconds = Column(Float, nullable=True)
    total_frames_processed = Column(Integer, nullable=True)
    
    # Processing status
    status = Column(String, default="uploaded")  # uploaded, processing, analyzed, completed
    processing_progress = Column(Float, default=0.0)  # 0 to 100
    
    # Weight measurements
    actual_weight_kg = Column(Float, nullable=True)  # From scale
    estimated_plastic_weight_kg = Column(Float, nullable=True)  # Calculated
    impurity_kg = Column(Float, nullable=True)
    impurity_percentage = Column(Float, nullable=True)
    
    # Counts
    total_bottles = Column(Integer, nullable=True)
    labeled_bottles = Column(Integer, nullable=True)
    pending_bottles = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    bottles = relationship("DetectedBottle", back_populates="batch")
    summary = relationship("BatchSummary", back_populates="batch", uselist=False)


class BatchSummary(Base):
    """Summary of bottle counts by label for a batch"""
    __tablename__ = "batch_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), unique=True)
    
    # JSON containing counts: {"Mojo 250mL": 245, "Pran 500mL": 132, ...}
    label_counts = Column(JSON, default={})
    
    # JSON containing weight estimates: {"Mojo 250mL": 2.94, ...}
    label_weights_kg = Column(JSON, default={})
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    batch = relationship("Batch", back_populates="summary")


# Note: Table creation is now handled in main.py after all models are imported