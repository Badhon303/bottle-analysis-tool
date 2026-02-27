# ============================================================================
# main.py - FastAPI Application Entry Point (v2)
# ============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict, field_validator
from datetime import datetime
from pathlib import Path
import shutil
import uuid
import numpy as np
import cv2

# Import configuration and database
from config import settings
from database import get_db, SessionLocal, Base, engine

# Import models to ensure tables are created at startup
# Note: batch must be imported first since bottle has FKs to batches table
from models.batch import Batch, BatchSummary
from models.bottle import BottleLabel, DetectedBottle

# Create all tables after all models are imported
Base.metadata.create_all(bind=engine)

# Import services
from services.detector import BottleDetector
from services.tracker import SimpleTracker
from services.feature_extractor import FeatureExtractor, FeatureExtractorCLIP, get_feature_extractor
from services.matcher import BottleMatcher
from services.clusterer import BottleClusterer
from services.video_processor import VideoProcessor

app = FastAPI(
    title="Bottle Vision System",
    description="Self-learning vision system for plastic bottle classification",
    version="1.0.0"
)

# CORS for frontend - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Serve cropped images
app.mount("/crops", StaticFiles(directory="crops"), name="crops")


# ============================================================================
# Pydantic Schemas
# ============================================================================

class LabelCreate(BaseModel):
    name: str
    brand: Optional[str] = None
    volume_ml: Optional[int] = None
    material: str = "PET"
    empty_weight_grams: Optional[float] = None
    description: Optional[str] = None

class LabelResponse(BaseModel):
    id: int
    name: str
    brand: Optional[str]
    volume_ml: Optional[int]
    material: str
    empty_weight_grams: Optional[float]
    sample_count: int = 0
    model_config = ConfigDict(from_attributes=True)
    
    @field_validator('sample_count', mode='before')
    @classmethod
    def ensure_sample_count(cls, v):
        return v if v is not None else 0

class BottleResponse(BaseModel):
    id: int
    batch_id: int
    track_id: int
    image_path: str
    label_id: Optional[int]
    label_name: Optional[str] = None
    confidence: Optional[float]
    status: str
    cluster_id: Optional[int]
    model_config = ConfigDict(from_attributes=True)

class BatchResponse(BaseModel):
    id: int
    name: str
    supplier: Optional[str]
    status: str
    processing_progress: float
    actual_weight_kg: Optional[float]
    estimated_plastic_weight_kg: Optional[float]
    impurity_kg: Optional[float]
    impurity_percentage: Optional[float]
    total_bottles: Optional[int]
    labeled_bottles: Optional[int]
    pending_bottles: Optional[int]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class BatchSummaryResponse(BaseModel):
    batch_id: int
    label_counts: dict
    label_weights_kg: dict
    total_bottles: int
    total_estimated_weight_kg: float

class ConfirmLabelRequest(BaseModel):
    bottle_ids: List[int]
    label_id: int

class CreateNewLabelRequest(BaseModel):
    bottle_ids: List[int]
    label: LabelCreate

class SetBatchWeightRequest(BaseModel):
    actual_weight_kg: float


# ============================================================================
# New Schemas for Image Analysis and Interactive Features
# ============================================================================

class SimilarityScore(BaseModel):
    """Similarity score between a bottle and a label"""
    label_id: int
    label_name: str
    similarity: float

class DetectedBottleInImage(BaseModel):
    """A bottle detected in an analyzed image"""
    id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    image_path: str
    label_id: Optional[int] = None
    label_name: Optional[str] = None
    match_confidence: Optional[float] = None
    status: str
    cluster_id: Optional[int] = None
    similarities: Optional[List[SimilarityScore]] = None

class ClusterInfo(BaseModel):
    """Information about a cluster of bottles"""
    cluster_id: int
    bottle_count: int
    bottle_ids: List[int]
    sample_images: List[str]

class ImageAnalysisResponse(BaseModel):
    """Response from image analysis endpoint"""
    image_id: str
    total_bottles: int
    matched_bottles: int
    unmatched_bottles: int
    bottles: List[DetectedBottleInImage]
    clusters: List[ClusterInfo]
    visualization_path: Optional[str] = None


class ImageBatchBottle(BaseModel):
    """A bottle detected in multi-image batch analysis"""
    id: int  # Global ID across all images in the batch
    source_image_index: int  # Which image this bottle came from (0-indexed)
    source_image_id: str  # The image_id of the source image
    local_id: int  # ID within the source image
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    image_path: str
    label_id: Optional[int] = None
    label_name: Optional[str] = None
    label_weight_grams: Optional[float] = None  # Weight of this bottle type in grams
    match_confidence: Optional[float] = None
    status: str
    cluster_id: Optional[int] = None
    similarities: Optional[List[SimilarityScore]] = None


class PerImageResult(BaseModel):
    """Results for a single image in a batch"""
    image_index: int
    image_id: str
    original_filename: str
    total_bottles: int
    matched_bottles: int
    unmatched_bottles: int
    estimated_weight_grams: Optional[float] = None  # Total weight of matched bottles in this image
    visualization_path: Optional[str] = None


class ImageBatchAnalysisResponse(BaseModel):
    """Response from multi-image batch analysis endpoint"""
    batch_id: str
    total_images: int
    total_bottles: int
    matched_bottles: int
    unmatched_bottles: int
    total_estimated_weight_grams: Optional[float] = None  # Total weight of all matched bottles
    total_estimated_weight_kg: Optional[float] = None  # Total weight in kg
    bottles: List[ImageBatchBottle]
    clusters: List[ClusterInfo]
    per_image_results: List[PerImageResult]

class ClusterDetailResponse(BaseModel):
    """Detailed information about a cluster"""
    cluster_id: int
    bottle_count: int
    bottles: List[DetectedBottleInImage]
    similarity_matrix: List[List[float]]
    suggested_labels: List[SimilarityScore]

class ReassignBottleRequest(BaseModel):
    """Request to reassign a bottle to a different label"""
    label_id: int
    source: str = "manual"  # "manual" or "correction"

class BatchReassignRequest(BaseModel):
    """Request to reassign multiple bottles"""
    bottle_ids: List[int]
    label_id: int
    source: str = "manual"


# ============================================================================
# Schemas for Image Analysis Label Creation
# ============================================================================

class CreateLabelFromImageRequest(BaseModel):
    """Create a new label from bottles in an image analysis"""
    image_id: str
    bottle_indices: List[int]  # Indices from the analysis response
    label: LabelCreate

class UpdateBottleClustersRequest(BaseModel):
    """Update cluster assignments for bottles in an image analysis"""
    image_id: str
    updates: List[Dict[str, int]]  # List of {"bottle_index": X, "new_cluster_id": Y}


class ConfigResponse(BaseModel):
    """Current system configuration"""
    yolo_confidence: float
    similarity_threshold: float
    high_confidence_threshold: float
    medium_confidence_threshold: float
    vit_model: str
    use_clip: bool
    clip_model: str
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int

class ConfigUpdate(BaseModel):
    """Update system configuration"""
    yolo_confidence: Optional[float] = None
    similarity_threshold: Optional[float] = None
    high_confidence_threshold: Optional[float] = None
    medium_confidence_threshold: Optional[float] = None
    vit_model: Optional[str] = None
    use_clip: Optional[bool] = None
    clip_model: Optional[str] = None
    hdbscan_min_cluster_size: Optional[int] = None
    hdbscan_min_samples: Optional[int] = None


# ============================================================================
# api/routes_labels.py - Label Management Endpoints
# ============================================================================

@app.get("/api/debug/ping")
def debug_ping():
    """Simple ping endpoint"""
    return {"status": "pong"}

@app.get("/api/debug/db-test")
def debug_db_test(db: Session = Depends(get_db)):
    """Debug endpoint to test database connection"""
    try:
        from sqlalchemy import text
        result = db.execute(text("SELECT 1")).scalar()
        return {"status": "ok", "db_test": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@app.get("/api/debug/labels-raw")
def debug_labels_raw(db: Session = Depends(get_db)):
    """Debug endpoint to test labels query without response model"""
    try:
        labels = db.query(BottleLabel).all()
        return {"status": "ok", "count": len(labels), "labels": [l.name for l in labels]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@app.post("/api/labels", response_model=LabelResponse)
def create_label(label: LabelCreate, db: Session = Depends(get_db)):
    """Create a new bottle label"""
    # Check if label already exists
    existing = db.query(BottleLabel).filter(BottleLabel.name == label.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Label already exists")
    
    db_label = BottleLabel(**label.dict())
    db.add(db_label)
    db.commit()
    db.refresh(db_label)
    
    return db_label

@app.get("/api/labels", response_model=List[LabelResponse])
def list_labels(db: Session = Depends(get_db)):
    """List all known bottle labels"""
    labels = db.query(BottleLabel).all()
    return labels

@app.get("/api/labels/{label_id}", response_model=LabelResponse)
def get_label(label_id: int, db: Session = Depends(get_db)):
    """Get a specific label"""
    label = db.query(BottleLabel).filter(BottleLabel.id == label_id).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    return label

@app.put("/api/labels/{label_id}", response_model=LabelResponse)
def update_label(label_id: int, label: LabelCreate, db: Session = Depends(get_db)):
    """Update a label"""
    db_label = db.query(BottleLabel).filter(BottleLabel.id == label_id).first()
    if not db_label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    for key, value in label.dict().items():
        setattr(db_label, key, value)
    
    db.commit()
    db.refresh(db_label)
    return db_label

@app.delete("/api/labels/{label_id}")
def delete_label(label_id: int, db: Session = Depends(get_db)):
    """Delete a label"""
    db_label = db.query(BottleLabel).filter(BottleLabel.id == label_id).first()
    if not db_label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    db.delete(db_label)
    db.commit()
    return {"message": "Label deleted"}


# ============================================================================
# api/routes_batch.py - Batch Processing Endpoints
# ============================================================================

@app.post("/api/batches", response_model=BatchResponse)
async def create_batch(
    video: UploadFile = File(...),
    supplier: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Upload a video and create a new batch for processing"""
    # Generate unique batch name
    batch_name = f"Batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    
    # Save video file
    video_filename = f"{batch_name}_{video.filename}"
    video_path = settings.UPLOAD_DIR / video_filename
    
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    # Create batch record
    batch = Batch(
        name=batch_name,
        supplier=supplier,
        video_path=str(video_path),
        status="uploaded"
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    
    # Start background processing
    background_tasks.add_task(process_batch_video, batch.id)
    
    return batch

@app.get("/api/batches", response_model=List[BatchResponse])
def list_batches(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all batches"""
    query = db.query(Batch)
    if status:
        query = query.filter(Batch.status == status)
    
    return query.order_by(Batch.created_at.desc()).limit(limit).all()

@app.get("/api/batches/{batch_id}", response_model=BatchResponse)
def get_batch(batch_id: int, db: Session = Depends(get_db)):
    """Get batch details"""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch

@app.get("/api/batches/{batch_id}/bottles", response_model=List[BottleResponse])
def get_batch_bottles(
    batch_id: int,
    status: Optional[str] = None,
    cluster_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get all bottles in a batch"""
    query = db.query(DetectedBottle).filter(DetectedBottle.batch_id == batch_id)
    
    if status:
        query = query.filter(DetectedBottle.status == status)
    if cluster_id is not None:
        query = query.filter(DetectedBottle.cluster_id == cluster_id)
    
    bottles = query.all()
    
    # Add label names
    result = []
    for bottle in bottles:
        bottle_dict = {
            "id": bottle.id,
            "batch_id": bottle.batch_id,
            "track_id": bottle.track_id,
            "image_path": f"/crops/{Path(bottle.image_path).name}",
            "label_id": bottle.label_id,
            "label_name": bottle.label.name if bottle.label else None,
            "confidence": bottle.confidence,
            "status": bottle.status,
            "cluster_id": bottle.cluster_id
        }
        result.append(bottle_dict)
    
    return result

@app.get("/api/batches/{batch_id}/clusters")
def get_batch_clusters(batch_id: int, db: Session = Depends(get_db)):
    """Get clusters of unknown bottles in a batch"""
    from sqlalchemy import func
    
    clusters = db.query(
        DetectedBottle.cluster_id,
        func.count(DetectedBottle.id).label('count')
    ).filter(
        DetectedBottle.batch_id == batch_id,
        DetectedBottle.status.in_(['pending', 'no_match']),
        DetectedBottle.cluster_id.isnot(None),
        DetectedBottle.cluster_id >= 0
    ).group_by(
        DetectedBottle.cluster_id
    ).all()
    
    result = []
    for cluster_id, count in clusters:
        # Get sample bottles from cluster
        samples = db.query(DetectedBottle).filter(
            DetectedBottle.batch_id == batch_id,
            DetectedBottle.cluster_id == cluster_id
        ).limit(5).all()
        
        result.append({
            "cluster_id": cluster_id,
            "count": count,
            "sample_images": [f"/crops/{Path(b.image_path).name}" for b in samples]
        })
    
    return result

@app.post("/api/batches/{batch_id}/weight")
def set_batch_weight(
    batch_id: int,
    request: SetBatchWeightRequest,
    db: Session = Depends(get_db)
):
    """Set the actual measured weight of the batch"""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch.actual_weight_kg = request.actual_weight_kg
    
    # Calculate impurity if estimated weight exists
    if batch.estimated_plastic_weight_kg:
        batch.impurity_kg = batch.actual_weight_kg - batch.estimated_plastic_weight_kg
        batch.impurity_percentage = (batch.impurity_kg / batch.actual_weight_kg) * 100
    
    db.commit()
    db.refresh(batch)
    
    return {
        "actual_weight_kg": batch.actual_weight_kg,
        "estimated_plastic_weight_kg": batch.estimated_plastic_weight_kg,
        "impurity_kg": batch.impurity_kg,
        "impurity_percentage": batch.impurity_percentage
    }

@app.get("/api/batches/{batch_id}/summary", response_model=BatchSummaryResponse)
def get_batch_summary(batch_id: int, db: Session = Depends(get_db)):
    """Get summary of bottle counts and weights by label"""
    from sqlalchemy import func
    
    # Count bottles by label
    counts = db.query(
        BottleLabel.name,
        BottleLabel.empty_weight_grams,
        func.count(DetectedBottle.id).label('count')
    ).join(
        DetectedBottle, DetectedBottle.label_id == BottleLabel.id
    ).filter(
        DetectedBottle.batch_id == batch_id,
        DetectedBottle.status.in_(['auto_labeled', 'confirmed'])
    ).group_by(
        BottleLabel.id
    ).all()
    
    label_counts = {}
    label_weights = {}
    total_weight = 0.0
    
    for name, weight_grams, count in counts:
        label_counts[name] = count
        if weight_grams:
            weight_kg = (weight_grams * count) / 1000
            label_weights[name] = round(weight_kg, 3)
            total_weight += weight_kg
    
    # Update batch estimated weight
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if batch:
        batch.estimated_plastic_weight_kg = round(total_weight, 3)
        batch.total_bottles = sum(label_counts.values())
        db.commit()
    
    return {
        "batch_id": batch_id,
        "label_counts": label_counts,
        "label_weights_kg": label_weights,
        "total_bottles": sum(label_counts.values()),
        "total_estimated_weight_kg": round(total_weight, 3)
    }


# ============================================================================
# api/routes_images.py - Single Image Analysis Endpoints
# ============================================================================

@app.post("/api/images/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    use_clip: bool = False,
    db: Session = Depends(get_db)
):
    """
    Analyze a single image for bottles.
    
    Detects all bottles, extracts features, matches against known labels,
    and clusters unmatched bottles.
    """
    # Generate unique image ID
    image_id = f"img-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    
    # Save uploaded image
    image_filename = f"{image_id}_{image.filename}"
    image_path = settings.UPLOAD_DIR / image_filename
    
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    
    # Load image with OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image file")
    
    # Initialize components
    detector = BottleDetector(
        model_path=settings.DETECTION_MODEL,
        confidence=settings.YOLO_CONFIDENCE
    )
    extractor = get_feature_extractor(
        use_clip=use_clip or settings.USE_CLIP,
        vit_model=settings.VIT_MODEL,
        clip_model=settings.CLIP_MODEL
    )
    clusterer = BottleClusterer(
        min_cluster_size=settings.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=settings.HDBSCAN_MIN_SAMPLES
    )
    matcher = BottleMatcher(
        high_threshold=settings.HIGH_CONFIDENCE_THRESHOLD,
        medium_threshold=settings.MEDIUM_CONFIDENCE_THRESHOLD
    )
    matcher.load_labels(db)
    
    # Step 1: Detect bottles
    detections = detector.detect(img)
    
    if not detections:
        return ImageAnalysisResponse(
            image_id=image_id,
            total_bottles=0,
            matched_bottles=0,
            unmatched_bottles=0,
            bottles=[],
            clusters=[],
            visualization_path=None
        )
    
    # Step 2: Extract features and match
    bottles = []
    matched_count = 0
    unmatched_embeddings = []
    unmatched_indices = []
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
        
        # Save crop
        crop_filename = f"{image_id}_bottle_{i}.jpg"
        crop_path = settings.CROPS_DIR / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        # Extract embedding
        embedding = extractor.extract(crop)
        
        # Match against known labels
        match_result = matcher.match(embedding)
        
        # Get all similarities for this bottle
        all_sims = matcher.get_all_similarities(embedding)
        similarities = [
            SimilarityScore(
                label_id=s["label_id"],
                label_name=s["label_name"],
                similarity=s["similarity"]
            )
            for s in all_sims[:5]  # Top 5
        ]
        
        if match_result.status == "high_confidence":
            status = "auto_labeled"
            matched_count += 1
        elif match_result.status == "medium_confidence":
            status = "pending"
        else:
            status = "no_match"
            unmatched_embeddings.append(embedding)
            unmatched_indices.append(i)
        
        bottle = DetectedBottleInImage(
            id=i,
            bbox=[x1, y1, x2, y2],
            confidence=det.confidence,
            image_path=f"/crops/{crop_filename}",
            label_id=match_result.label_id,
            label_name=match_result.label_name,
            match_confidence=match_result.confidence,
            status=status,
            cluster_id=None,
            similarities=similarities
        )
        bottles.append(bottle)
    
    # Step 3: Cluster unmatched bottles
    clusters = []
    if unmatched_embeddings:
        embeddings_array = np.array(unmatched_embeddings)
        cluster_ids = clusterer.cluster(embeddings_array, list(range(len(unmatched_embeddings))))
        
        # Group by cluster
        cluster_groups: Dict[int, List[int]] = {}
        for idx, cluster_id in cluster_ids.items():
            original_idx = unmatched_indices[idx]
            bottles[original_idx].cluster_id = cluster_id
            
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(original_idx)
        
        # Create cluster info
        for cid, bottle_indices in cluster_groups.items():
            clusters.append(ClusterInfo(
                cluster_id=cid,
                bottle_count=len(bottle_indices),
                bottle_ids=bottle_indices,
                sample_images=[bottles[bi].image_path for bi in bottle_indices[:5]]
            ))
    
    # Step 4: Generate visualization
    output = img.copy()
    colors = {}
    
    for bottle in bottles:
        label = bottle.label_name or f"Cluster {bottle.cluster_id}" if bottle.cluster_id is not None else "Unknown"
        if label not in colors:
            colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
        
        x1, y1, x2, y2 = bottle.bbox
        color = colors[label]
        
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label[:15]} (#{bottle.id})"
        cv2.putText(output, label_text, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    viz_filename = f"{image_id}_visualization.jpg"
    viz_path = settings.CROPS_DIR / viz_filename
    cv2.imwrite(str(viz_path), output)
    
    return ImageAnalysisResponse(
        image_id=image_id,
        total_bottles=len(bottles),
        matched_bottles=matched_count,
        unmatched_bottles=len(bottles) - matched_count,
        bottles=bottles,
        clusters=clusters,
        visualization_path=f"/crops/{viz_filename}"
    )


# ============================================================================
# Multi-Image Batch Analysis Endpoint
# ============================================================================

@app.post("/api/images/analyze-batch", response_model=ImageBatchAnalysisResponse)
async def analyze_image_batch(
    images: List[UploadFile] = File(...),
    use_clip: bool = False,
    db: Session = Depends(get_db)
):
    """
    Analyze multiple images for bottles in a single request.
    
    Detects bottles in all images, extracts features, matches against known labels,
    and clusters unmatched bottles ACROSS ALL IMAGES. This means similar bottles
    from different images will be grouped together in the same cluster.
    
    Returns:
    - Combined totals across all images
    - All detected bottles with their source image info
    - Clusters of unmatched bottles (spanning all images)
    - Per-image breakdown of results
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(images) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images allowed per batch")
    
    # Generate unique batch ID
    batch_id = f"img-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    
    # Initialize components (shared across all images for efficiency)
    detector = BottleDetector(
        model_path=settings.DETECTION_MODEL,
        confidence=settings.YOLO_CONFIDENCE
    )
    extractor = get_feature_extractor(
        use_clip=use_clip or settings.USE_CLIP,
        vit_model=settings.VIT_MODEL,
        clip_model=settings.CLIP_MODEL
    )
    clusterer = BottleClusterer(
        min_cluster_size=settings.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=settings.HDBSCAN_MIN_SAMPLES
    )
    matcher = BottleMatcher(
        high_threshold=settings.HIGH_CONFIDENCE_THRESHOLD,
        medium_threshold=settings.MEDIUM_CONFIDENCE_THRESHOLD
    )
    matcher.load_labels(db)
    
    # Process all images
    all_bottles: List[ImageBatchBottle] = []
    per_image_results: List[PerImageResult] = []
    all_unmatched_embeddings = []
    all_unmatched_global_indices = []
    global_bottle_id = 0
    total_matched = 0
    
    for img_idx, uploaded_image in enumerate(images):
        # Generate unique image ID
        image_id = f"{batch_id}_img{img_idx}_{uuid.uuid4().hex[:4]}"
        
        # Save uploaded image
        image_filename = f"{image_id}_{uploaded_image.filename}"
        image_path = settings.UPLOAD_DIR / image_filename
        
        with open(image_path, "wb") as f:
            shutil.copyfileobj(uploaded_image.file, f)
        
        # Load image with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            # Skip unreadable images but continue processing
            per_image_results.append(PerImageResult(
                image_index=img_idx,
                image_id=image_id,
                original_filename=uploaded_image.filename,
                total_bottles=0,
                matched_bottles=0,
                unmatched_bottles=0,
                visualization_path=None
            ))
            continue
        
        # Detect bottles
        detections = detector.detect(img)
        
        image_bottles = []
        image_matched = 0
        image_unmatched_indices = []
        image_weight_grams = 0.0  # Track weight for this image
        
        for local_id, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Save crop
            crop_filename = f"{image_id}_bottle_{local_id}.jpg"
            crop_path = settings.CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)
            
            # Extract embedding
            embedding = extractor.extract(crop)
            
            # Match against known labels
            match_result = matcher.match(embedding)
            
            # Get all similarities for this bottle
            all_sims = matcher.get_all_similarities(embedding)
            similarities = [
                SimilarityScore(
                    label_id=s["label_id"],
                    label_name=s["label_name"],
                    similarity=s["similarity"]
                )
                for s in all_sims[:5]  # Top 5
            ]
            
            if match_result.status == "high_confidence":
                status = "auto_labeled"
                image_matched += 1
                total_matched += 1
                # Add weight if available
                if match_result.empty_weight_grams:
                    image_weight_grams += match_result.empty_weight_grams
            elif match_result.status == "medium_confidence":
                status = "pending"
            else:
                status = "no_match"
                all_unmatched_embeddings.append(embedding)
                all_unmatched_global_indices.append(global_bottle_id)
                image_unmatched_indices.append(global_bottle_id)
            
            bottle = ImageBatchBottle(
                id=global_bottle_id,
                source_image_index=img_idx,
                source_image_id=image_id,
                local_id=local_id,
                bbox=[x1, y1, x2, y2],
                confidence=det.confidence,
                image_path=f"/crops/{crop_filename}",
                label_id=match_result.label_id,
                label_name=match_result.label_name,
                label_weight_grams=match_result.empty_weight_grams,
                match_confidence=match_result.confidence,
                status=status,
                cluster_id=None,
                similarities=similarities
            )
            image_bottles.append(bottle)
            all_bottles.append(bottle)
            global_bottle_id += 1
        
        # Generate per-image visualization
        if detections:
            output = img.copy()
            colors = {}
            
            for bottle in image_bottles:
                label = bottle.label_name or f"Unmatched" if bottle.cluster_id is None else f"Cluster {bottle.cluster_id}"
                if label not in colors:
                    colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
                
                x1, y1, x2, y2 = bottle.bbox
                color = colors[label]
                
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label[:15]} (#{bottle.local_id})"
                cv2.putText(output, label_text, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            viz_filename = f"{image_id}_visualization.jpg"
            viz_path = settings.CROPS_DIR / viz_filename
            cv2.imwrite(str(viz_path), output)
            viz_path_str = f"/crops/{viz_filename}"
        else:
            viz_path_str = None
        
        per_image_results.append(PerImageResult(
            image_index=img_idx,
            image_id=image_id,
            original_filename=uploaded_image.filename,
            total_bottles=len(image_bottles),
            matched_bottles=image_matched,
            unmatched_bottles=len(image_bottles) - image_matched,
            estimated_weight_grams=round(image_weight_grams, 2) if image_weight_grams > 0 else None,
            visualization_path=viz_path_str
        ))
    
    # Calculate total weight across all images
    total_weight_grams = sum(
        bottle.label_weight_grams for bottle in all_bottles 
        if bottle.label_weight_grams is not None and bottle.status == "auto_labeled"
    )
    
    # Cluster ALL unmatched bottles across all images
    clusters = []
    if all_unmatched_embeddings:
        embeddings_array = np.array(all_unmatched_embeddings)
        cluster_ids = clusterer.cluster(embeddings_array, list(range(len(all_unmatched_embeddings))))
        
        # Group by cluster
        cluster_groups: Dict[int, List[int]] = {}
        for idx, cluster_id in cluster_ids.items():
            global_idx = all_unmatched_global_indices[idx]
            
            # Update the bottle's cluster_id
            all_bottles[global_idx].cluster_id = cluster_id
            
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(global_idx)
        
        # Create cluster info
        for cid, bottle_indices in cluster_groups.items():
            clusters.append(ClusterInfo(
                cluster_id=cid,
                bottle_count=len(bottle_indices),
                bottle_ids=bottle_indices,
                sample_images=[all_bottles[bi].image_path for bi in bottle_indices[:5]]
            ))
    
    return ImageBatchAnalysisResponse(
        batch_id=batch_id,
        total_images=len(images),
        total_bottles=len(all_bottles),
        matched_bottles=total_matched,
        unmatched_bottles=len(all_bottles) - total_matched,
        total_estimated_weight_grams=round(total_weight_grams, 2) if total_weight_grams > 0 else None,
        total_estimated_weight_kg=round(total_weight_grams / 1000, 4) if total_weight_grams > 0 else None,
        bottles=all_bottles,
        clusters=clusters,
        per_image_results=per_image_results
    )


# ============================================================================
# Image Analysis - Label Creation from Detected Bottles
# ============================================================================

@app.post("/api/images/create-label")
async def create_label_from_image_analysis(
    request: CreateLabelFromImageRequest,
    use_clip: bool = False,
    db: Session = Depends(get_db)
):
    """
    Create a new label from selected bottles in an image analysis.
    
    This endpoint allows you to select bottle indices from a previous /api/images/analyze
    response and create a new label with their averaged embedding.
    
    The crops must still exist in the crops directory (they're saved during analysis).
    """
    # Check if label name already exists
    existing = db.query(BottleLabel).filter(BottleLabel.name == request.label.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Label '{request.label.name}' already exists")
    
    # Initialize feature extractor
    extractor = get_feature_extractor(
        use_clip=use_clip or settings.USE_CLIP,
        vit_model=settings.VIT_MODEL,
        clip_model=settings.CLIP_MODEL
    )
    
    # Load embeddings from crop images
    embeddings = []
    valid_indices = []
    
    for idx in request.bottle_indices:
        crop_filename = f"{request.image_id}_bottle_{idx}.jpg"
        crop_path = settings.CROPS_DIR / crop_filename
        
        if not crop_path.exists():
            continue
        
        # Load and extract embedding
        crop_img = cv2.imread(str(crop_path))
        if crop_img is None:
            continue
        
        embedding = extractor.extract(crop_img)
        embeddings.append(embedding)
        valid_indices.append(idx)
    
    if not embeddings:
        raise HTTPException(
            status_code=400, 
            detail=f"No valid crops found for image_id '{request.image_id}'. Make sure the crops still exist."
        )
    
    # Calculate average embedding
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Create label
    db_label = BottleLabel(**request.label.dict())
    db_label.reference_embedding = avg_embedding.tolist()
    db_label.sample_count = len(embeddings)
    
    db.add(db_label)
    db.commit()
    db.refresh(db_label)
    
    return {
        "message": f"Created label '{db_label.name}' from {len(embeddings)} bottles",
        "label_id": db_label.id,
        "label_name": db_label.name,
        "sample_count": len(embeddings),
        "bottle_indices_used": valid_indices
    }


@app.post("/api/images/add-to-label")
async def add_bottles_to_existing_label(
    image_id: str,
    bottle_indices: List[int],
    label_id: int,
    use_clip: bool = False,
    db: Session = Depends(get_db)
):
    """
    Add bottles from an image analysis to an existing label.
    
    This updates the label's reference embedding using a running average.
    """
    # Get label
    label = db.query(BottleLabel).filter(BottleLabel.id == label_id).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    # Initialize feature extractor
    extractor = get_feature_extractor(
        use_clip=use_clip or settings.USE_CLIP,
        vit_model=settings.VIT_MODEL,
        clip_model=settings.CLIP_MODEL
    )
    
    # Initialize matcher for updating embeddings
    matcher = BottleMatcher()
    
    added_count = 0
    for idx in bottle_indices:
        crop_filename = f"{image_id}_bottle_{idx}.jpg"
        crop_path = settings.CROPS_DIR / crop_filename
        
        if not crop_path.exists():
            continue
        
        crop_img = cv2.imread(str(crop_path))
        if crop_img is None:
            continue
        
        embedding = extractor.extract(crop_img)
        matcher.update_label_embedding(db, label_id, embedding)
        added_count += 1
    
    if added_count == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No valid crops found for image_id '{image_id}'"
        )
    
    # Refresh label to get updated sample count
    db.refresh(label)
    
    return {
        "message": f"Added {added_count} bottles to label '{label.name}'",
        "label_id": label.id,
        "new_sample_count": label.sample_count
    }


@app.get("/api/images/{image_id}/bottles")
async def get_image_bottles(image_id: str):
    """
    Get all bottle crops for a previously analyzed image.
    
    Returns the list of available crops and their paths.
    """
    crops = list(settings.CROPS_DIR.glob(f"{image_id}_bottle_*.jpg"))
    
    if not crops:
        raise HTTPException(
            status_code=404,
            detail=f"No crops found for image_id '{image_id}'"
        )
    
    # Sort by bottle index
    def get_index(path):
        name = path.stem
        parts = name.split("_bottle_")
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                return 999
        return 999
    
    crops.sort(key=get_index)
    
    return {
        "image_id": image_id,
        "bottle_count": len(crops),
        "bottles": [
            {
                "index": get_index(crop),
                "image_path": f"/crops/{crop.name}"
            }
            for crop in crops
        ]
    }


# ============================================================================
# api/routes_bottles.py - Bottle Management Endpoints
# ============================================================================

@app.post("/api/bottles/confirm")
def confirm_bottle_labels(request: ConfirmLabelRequest, db: Session = Depends(get_db)):
    """Confirm labels for multiple bottles (operator approval)"""
    label = db.query(BottleLabel).filter(BottleLabel.id == request.label_id).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    # Update bottles
    bottles = db.query(DetectedBottle).filter(
        DetectedBottle.id.in_(request.bottle_ids)
    ).all()
    
    # Initialize matcher to update embeddings
    matcher = BottleMatcher()
    
    for bottle in bottles:
        bottle.label_id = request.label_id
        bottle.status = "confirmed"
        
        # Update label's reference embedding with this sample
        if bottle.embedding:
            matcher.update_label_embedding(db, request.label_id, np.array(bottle.embedding))
    
    db.commit()
    
    return {"message": f"Confirmed {len(bottles)} bottles as {label.name}"}

@app.post("/api/bottles/new-label")
def create_label_from_bottles(request: CreateNewLabelRequest, db: Session = Depends(get_db)):
    """Create a new label from a cluster of unknown bottles"""
    # Create new label
    db_label = BottleLabel(**request.label.dict())
    db.add(db_label)
    db.commit()
    db.refresh(db_label)
    
    # Update bottles
    bottles = db.query(DetectedBottle).filter(
        DetectedBottle.id.in_(request.bottle_ids)
    ).all()
    
    # Calculate average embedding for new label
    embeddings = [np.array(b.embedding) for b in bottles if b.embedding]
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        db_label.reference_embedding = avg_embedding.tolist()
        db_label.sample_count = len(embeddings)
    
    # Assign label to bottles
    for bottle in bottles:
        bottle.label_id = db_label.id
        bottle.status = "confirmed"
    
    db.commit()
    
    return {
        "message": f"Created new label '{db_label.name}' with {len(bottles)} bottles",
        "label_id": db_label.id
    }

@app.post("/api/bottles/{bottle_id}/assign-label")
def assign_label_to_bottle(
    bottle_id: int,
    label_id: int,
    db: Session = Depends(get_db)
):
    """Assign a label to a single bottle"""
    bottle = db.query(DetectedBottle).filter(DetectedBottle.id == bottle_id).first()
    if not bottle:
        raise HTTPException(status_code=404, detail="Bottle not found")
    
    label = db.query(BottleLabel).filter(BottleLabel.id == label_id).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    bottle.label_id = label_id
    bottle.status = "confirmed"
    
    # Update label embedding
    if bottle.embedding:
        matcher = BottleMatcher()
        matcher.update_label_embedding(db, label_id, np.array(bottle.embedding))
    
    db.commit()
    
    return {"message": f"Assigned bottle to {label.name}"}


@app.get("/api/bottles/{bottle_id}/similarities", response_model=List[SimilarityScore])
def get_bottle_similarities(bottle_id: int, db: Session = Depends(get_db)):
    """Get similarity scores between a bottle and all known labels"""
    bottle = db.query(DetectedBottle).filter(DetectedBottle.id == bottle_id).first()
    if not bottle:
        raise HTTPException(status_code=404, detail="Bottle not found")
    
    if not bottle.embedding:
        raise HTTPException(status_code=400, detail="Bottle has no embedding")
    
    # Initialize matcher and load labels
    matcher = BottleMatcher(
        high_threshold=settings.HIGH_CONFIDENCE_THRESHOLD,
        medium_threshold=settings.MEDIUM_CONFIDENCE_THRESHOLD
    )
    matcher.load_labels(db)
    
    # Get all similarities
    embedding = np.array(bottle.embedding)
    similarities = matcher.get_all_similarities(embedding)
    
    return [
        SimilarityScore(
            label_id=s["label_id"],
            label_name=s["label_name"],
            similarity=s["similarity"]
        )
        for s in similarities
    ]


@app.post("/api/bottles/{bottle_id}/reassign")
def reassign_bottle(
    bottle_id: int,
    request: ReassignBottleRequest,
    db: Session = Depends(get_db)
):
    """Reassign a bottle to a different label (manual correction)"""
    bottle = db.query(DetectedBottle).filter(DetectedBottle.id == bottle_id).first()
    if not bottle:
        raise HTTPException(status_code=404, detail="Bottle not found")
    
    label = db.query(BottleLabel).filter(BottleLabel.id == request.label_id).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    old_label = bottle.label.name if bottle.label else "Unknown"
    
    bottle.label_id = request.label_id
    bottle.status = "confirmed"
    
    # Update label embedding with correction
    if bottle.embedding:
        matcher = BottleMatcher()
        matcher.update_label_embedding(db, request.label_id, np.array(bottle.embedding))
    
    db.commit()
    
    return {
        "message": f"Reassigned bottle from '{old_label}' to '{label.name}'",
        "source": request.source
    }


@app.post("/api/bottles/batch-reassign")
def batch_reassign_bottles(request: BatchReassignRequest, db: Session = Depends(get_db)):
    """Reassign multiple bottles to a label (batch operation)"""
    label = db.query(BottleLabel).filter(BottleLabel.id == request.label_id).first()
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")
    
    bottles = db.query(DetectedBottle).filter(
        DetectedBottle.id.in_(request.bottle_ids)
    ).all()
    
    if not bottles:
        raise HTTPException(status_code=404, detail="No bottles found")
    
    matcher = BottleMatcher()
    
    for bottle in bottles:
        bottle.label_id = request.label_id
        bottle.status = "confirmed"
        
        if bottle.embedding:
            matcher.update_label_embedding(db, request.label_id, np.array(bottle.embedding))
    
    db.commit()
    
    return {
        "message": f"Reassigned {len(bottles)} bottles to '{label.name}'",
        "source": request.source
    }


@app.get("/api/batches/{batch_id}/clusters/{cluster_id}/details", response_model=ClusterDetailResponse)
def get_cluster_details(batch_id: int, cluster_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific cluster"""
    bottles = db.query(DetectedBottle).filter(
        DetectedBottle.batch_id == batch_id,
        DetectedBottle.cluster_id == cluster_id
    ).all()
    
    if not bottles:
        raise HTTPException(status_code=404, detail="Cluster not found or empty")
    
    # Initialize matcher for similarity scoring
    matcher = BottleMatcher(
        high_threshold=settings.HIGH_CONFIDENCE_THRESHOLD,
        medium_threshold=settings.MEDIUM_CONFIDENCE_THRESHOLD
    )
    matcher.load_labels(db)
    
    # Build bottle responses with similarities
    bottle_responses = []
    embeddings = []
    
    for bottle in bottles:
        embedding = np.array(bottle.embedding) if bottle.embedding else None
        
        similarities = []
        if embedding is not None:
            embeddings.append(embedding)
            sims = matcher.get_all_similarities(embedding)
            similarities = [
                SimilarityScore(
                    label_id=s["label_id"],
                    label_name=s["label_name"],
                    similarity=s["similarity"]
                )
                for s in sims[:5]
            ]
        
        bottle_responses.append(DetectedBottleInImage(
            id=bottle.id,
            bbox=[0, 0, 0, 0],  # Bbox not stored in DB for tracked bottles
            confidence=bottle.confidence or 0.0,
            image_path=f"/crops/{Path(bottle.image_path).name}",
            label_id=bottle.label_id,
            label_name=bottle.label.name if bottle.label else None,
            match_confidence=bottle.confidence,
            status=bottle.status,
            cluster_id=bottle.cluster_id,
            similarities=similarities
        ))
    
    # Compute similarity matrix
    similarity_matrix = []
    if embeddings:
        sim_matrix = matcher.compute_similarity_matrix(embeddings)
        similarity_matrix = sim_matrix.tolist()
    
    # Get suggested labels (average of all bottle similarities)
    suggested_labels = []
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        sims = matcher.get_all_similarities(avg_embedding)
        suggested_labels = [
            SimilarityScore(
                label_id=s["label_id"],
                label_name=s["label_name"],
                similarity=s["similarity"]
            )
            for s in sims[:5]
        ]
    
    return ClusterDetailResponse(
        cluster_id=cluster_id,
        bottle_count=len(bottles),
        bottles=bottle_responses,
        similarity_matrix=similarity_matrix,
        suggested_labels=suggested_labels
    )


# ============================================================================
# api/routes_config.py - System Configuration Endpoints
# ============================================================================

# Runtime configuration (in-memory, resets on restart)
_runtime_config = {
    "yolo_confidence": settings.YOLO_CONFIDENCE,
    "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    "high_confidence_threshold": settings.HIGH_CONFIDENCE_THRESHOLD,
    "medium_confidence_threshold": settings.MEDIUM_CONFIDENCE_THRESHOLD,
    "vit_model": settings.VIT_MODEL,
    "use_clip": settings.USE_CLIP,
    "clip_model": settings.CLIP_MODEL,
    "hdbscan_min_cluster_size": settings.HDBSCAN_MIN_CLUSTER_SIZE,
    "hdbscan_min_samples": settings.HDBSCAN_MIN_SAMPLES,
}


@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    """Get current system configuration"""
    return ConfigResponse(**_runtime_config)


@app.put("/api/config", response_model=ConfigResponse)
def update_config(config: ConfigUpdate):
    """Update system configuration (runtime only, resets on restart)"""
    if config.yolo_confidence is not None:
        if not 0.0 <= config.yolo_confidence <= 1.0:
            raise HTTPException(status_code=400, detail="yolo_confidence must be between 0 and 1")
        _runtime_config["yolo_confidence"] = config.yolo_confidence
    
    if config.similarity_threshold is not None:
        if not 0.0 <= config.similarity_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="similarity_threshold must be between 0 and 1")
        _runtime_config["similarity_threshold"] = config.similarity_threshold
    
    if config.high_confidence_threshold is not None:
        if not 0.0 <= config.high_confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="high_confidence_threshold must be between 0 and 1")
        _runtime_config["high_confidence_threshold"] = config.high_confidence_threshold
    
    if config.medium_confidence_threshold is not None:
        if not 0.0 <= config.medium_confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="medium_confidence_threshold must be between 0 and 1")
        _runtime_config["medium_confidence_threshold"] = config.medium_confidence_threshold
    
    if config.vit_model is not None:
        valid_models = ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]
        if config.vit_model not in valid_models:
            raise HTTPException(status_code=400, detail=f"vit_model must be one of {valid_models}")
        _runtime_config["vit_model"] = config.vit_model
    
    if config.use_clip is not None:
        _runtime_config["use_clip"] = config.use_clip
    
    if config.clip_model is not None:
        valid_clip = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]
        if config.clip_model not in valid_clip:
            raise HTTPException(status_code=400, detail=f"clip_model must be one of {valid_clip}")
        _runtime_config["clip_model"] = config.clip_model
    
    if config.hdbscan_min_cluster_size is not None:
        if config.hdbscan_min_cluster_size < 2:
            raise HTTPException(status_code=400, detail="hdbscan_min_cluster_size must be at least 2")
        _runtime_config["hdbscan_min_cluster_size"] = config.hdbscan_min_cluster_size
    
    if config.hdbscan_min_samples is not None:
        if config.hdbscan_min_samples < 1:
            raise HTTPException(status_code=400, detail="hdbscan_min_samples must be at least 1")
        _runtime_config["hdbscan_min_samples"] = config.hdbscan_min_samples
    
    return ConfigResponse(**_runtime_config)


# ============================================================================
# Background Processing Task
# ============================================================================

def process_batch_video(batch_id: int):
    """Background task to process a batch video"""
    db = SessionLocal()
    
    try:
        batch = db.query(Batch).filter(Batch.id == batch_id).first()
        if not batch:
            return
        
        batch.status = "processing"
        db.commit()
        
        # Initialize components with runtime config
        detector = BottleDetector(
            model_path=settings.DETECTION_MODEL,
            confidence=_runtime_config["yolo_confidence"]
        )
        tracker = SimpleTracker()
        extractor = get_feature_extractor(
            use_clip=_runtime_config["use_clip"],
            vit_model=_runtime_config["vit_model"],
            clip_model=_runtime_config["clip_model"]
        )
        matcher = BottleMatcher(
            high_threshold=_runtime_config["high_confidence_threshold"],
            medium_threshold=_runtime_config["medium_confidence_threshold"]
        )
        matcher.load_labels(db)
        
        # Process video
        with VideoProcessor(batch.video_path, target_fps=settings.FRAME_RATE) as vp:
            batch.video_duration_seconds = vp.duration_seconds
            total_frames = vp.total_frames // max(1, int(vp.video_fps / settings.FRAME_RATE))
            
            for i, frame_data in enumerate(vp.extract_frames()):
                # Detect bottles
                detections = detector.detect(frame_data.image)
                
                # Track across frames
                tracker.update(frame_data.frame_number, detections, frame_data.image)
                
                # Update progress
                batch.processing_progress = (i / total_frames) * 100
                db.commit()
        
        # Process completed tracks
        batch.total_frames_processed = total_frames
        
        detected_bottles = []
        unknown_embeddings = []
        unknown_bottle_ids = []
        
        for track in tracker.get_all_tracks():
            if len(track.detections) < 2:  # Skip very brief detections
                continue
            
            # Get best crop
            frame_num, crop, sharpness = track.get_best_crop()
            if crop is None or crop.size == 0:
                continue
            
            # Save crop image
            crop_filename = f"{batch.name}_track{track.track_id}.jpg"
            crop_path = settings.CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)
            
            # Extract embedding
            embedding = extractor.extract(crop)
            
            # Match to known labels
            match_result = matcher.match(embedding)
            
            # Create bottle record
            bottle = DetectedBottle(
                batch_id=batch_id,
                track_id=track.track_id,
                image_path=str(crop_path),
                best_frame_number=frame_num,
                sharpness_score=sharpness,
                embedding=embedding.tolist(),
                label_id=match_result.label_id,
                confidence=match_result.confidence,
                status="auto_labeled" if match_result.status == "high_confidence"
                       else "pending" if match_result.status == "medium_confidence"
                       else "no_match"
            )
            db.add(bottle)
            db.flush()  # Get the ID
            
            detected_bottles.append(bottle)
            
            # Track unknowns for clustering
            if match_result.status == "no_match":
                unknown_embeddings.append(embedding)
                unknown_bottle_ids.append(bottle.id)
        
        # Cluster unknown bottles
        if unknown_embeddings:
            clusterer = BottleClusterer(
                min_cluster_size=_runtime_config["hdbscan_min_cluster_size"],
                min_samples=_runtime_config["hdbscan_min_samples"]
            )

            cluster_map = clusterer.cluster(
                np.array(unknown_embeddings),
                unknown_bottle_ids
            )
            
            for bottle_id, cluster_id in cluster_map.items():
                bottle = db.query(DetectedBottle).filter(
                    DetectedBottle.id == bottle_id
                ).first()
                if bottle:
                    bottle.cluster_id = cluster_id
        
        # Update batch stats
        batch.total_bottles = len(detected_bottles)
        batch.labeled_bottles = len([b for b in detected_bottles if b.status == "auto_labeled"])
        batch.pending_bottles = len([b for b in detected_bottles if b.status in ["pending", "no_match"]])
        batch.status = "analyzed"
        batch.processing_progress = 100.0
        
        db.commit()
        
    except Exception as e:
        batch.status = f"error: {str(e)}"
        db.commit()
        raise
    finally:
        db.close()


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8066, reload=True)