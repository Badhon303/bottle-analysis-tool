# Bottle Vision System API Documentation

Base URL: `http://localhost:8066`

## Overview

The Bottle Vision System API provides endpoints for:
- Managing bottle labels
- Processing video batches
- Analyzing single/multiple images
- Managing individual bottle detections and classifications
- System configuration

## Authentication

Currently, the API does not enforce authentication.

## Endpoints

### 1. General / Debug

#### `GET /api/debug/ping`
Simple ping to check if the server is running.
- **Response**: `{"status": "pong"}`

#### `GET /api/debug/db-test`
Tests the database connection.
- **Response**: `{"status": "ok", "db_test": 1}` or error details.

#### `GET /api/debug/labels-raw`
Returns a raw list of label names from the database (bypassing Pydantic models).
- **Response**: `{"status": "ok", "count": <int>, "labels": [<str>, ...]}`

---

### 2. Label Management

#### `POST /api/labels`
Create a new bottle label.
- **Request Body** (`LabelCreate`):
  ```json
  {
    "name": "string",
    "brand": "string (optional)",
    "volume_ml": "integer (optional)",
    "material": "string (default: 'PET')",
    "empty_weight_grams": "float (optional)",
    "description": "string (optional)"
  }
  ```
- **Response** (`LabelResponse`):
  ```json
  {
    "id": "integer",
    "name": "string",
    "brand": "string",
    "volume_ml": "integer",
    "material": "string",
    "empty_weight_grams": "float",
    "sample_count": "integer"
  }
  ```

#### `GET /api/labels`
List all known bottle labels.
- **Response**: List of `LabelResponse` objects.

#### `GET /api/labels/{label_id}`
Get details of a specific label.
- **Path Parameters**: `label_id` (integer)
- **Response**: `LabelResponse` object.

#### `PUT /api/labels/{label_id}`
Update an existing label.
- **Path Parameters**: `label_id` (integer)
- **Request Body**: `LabelCreate` object.
- **Response**: Updated `LabelResponse` object.

#### `DELETE /api/labels/{label_id}`
Delete a label.
- **Path Parameters**: `label_id` (integer)
- **Response**: `{"message": "Label deleted"}`

---

### 3. Batch Processing (Video)

#### `POST /api/batches`
Upload a video and create a new batch for processing. Processing happens in the background.
- **Request Body (Multipart)**:
  - `video`: File (required)
  - `supplier`: String (optional)
- **Response** (`BatchResponse`):
  ```json
  {
    "id": "integer",
    "name": "string",
    "supplier": "string",
    "status": "uploaded",
    "processing_progress": 0.0,
    "created_at": "datetime"
    ...
  }
  ```

#### `GET /api/batches`
List all batches.
- **Query Parameters**:
  - `status`: String (optional, e.g., 'processing', 'analyzed')
  - `limit`: Integer (default: 50)
- **Response**: List of `BatchResponse` objects.

#### `GET /api/batches/{batch_id}`
Get details of a specific batch.
- **Path Parameters**: `batch_id` (integer)
- **Response**: `BatchResponse` object.

#### `GET /api/batches/{batch_id}/bottles`
Get all bottles detected in a batch.
- **Path Parameters**: `batch_id` (integer)
- **Query Parameters**:
  - `status`: String (optional, e.g., 'auto_labeled', 'pending', 'no_match')
  - `cluster_id`: Integer (optional)
- **Response**: List of `BottleResponse` objects.
  ```json
  {
    "id": "integer",
    "batch_id": "integer",
    "track_id": "integer",
    "image_path": "string (URL)",
    "label_id": "integer (optional)",
    "label_name": "string (optional)",
    "confidence": "float",
    "status": "string",
    "cluster_id": "integer (optional)"
  }
  ```

#### `GET /api/batches/{batch_id}/clusters`
Get clusters of unknown bottles in a batch.
- **Path Parameters**: `batch_id` (integer)
- **Response**: List of cluster summaries.
  ```json
  [
    {
      "cluster_id": "integer",
      "count": "integer",
      "sample_images": ["string (URL)", ...]
    }
  ]
  ```

#### `GET /api/batches/{batch_id}/clusters/{cluster_id}/details`
Get detailed information about a specific cluster, including similarity matrix and suggested labels.
- **Path Parameters**:
  - `batch_id`: Integer
  - `cluster_id`: Integer
- **Response** (`ClusterDetailResponse`):
  ```json
  {
    "cluster_id": "integer",
    "bottle_count": "integer",
    "bottles": [ ... ],
    "similarity_matrix": [[float, ...], ...],
    "suggested_labels": [ ... ]
  }
  ```

#### `POST /api/batches/{batch_id}/weight`
Set the actual measured weight of the batch for impurity calculation.
- **Path Parameters**: `batch_id` (integer)
- **Request Body**:
  ```json
  {
    "actual_weight_kg": "float"
  }
  ```
- **Response**:
  ```json
  {
    "actual_weight_kg": "float",
    "estimated_plastic_weight_kg": "float",
    "impurity_kg": "float",
    "impurity_percentage": "float"
  }
  ```

#### `GET /api/batches/{batch_id}/summary`
Get a summary of the batch including bottle counts and weights by label.
- **Path Parameters**: `batch_id` (integer)
- **Response** (`BatchSummaryResponse`):
  ```json
  {
    "batch_id": "integer",
    "label_counts": { "Label Name": count, ... },
    "label_weights_kg": { "Label Name": weight, ... },
    "total_bottles": "integer",
    "total_estimated_weight_kg": "float"
  }
  ```

---

### 4. Image Analysis

#### `POST /api/images/analyze`
Analyze a single image to detect and classify bottles.
- **Request Body (Multipart)**:
  - `image`: File (required)
  - `use_clip`: Boolean (optional, default: False)
- **Response** (`ImageAnalysisResponse`):
  ```json
  {
    "image_id": "string",
    "total_bottles": "integer",
    "matched_bottles": "integer",
    "unmatched_bottles": "integer",
    "bottles": [ ... ],
    "clusters": [ ... ],
    "visualization_path": "string (URL)"
  }
  ```

#### `POST /api/images/analyze-batch`
Analyze multiple images in a single request and cluster unmatched bottles across all images.
- **Request Body (Multipart)**:
  - `images`: List of files (required, max 50)
  - `use_clip`: Boolean (optional, default: False)
- **Response** (`ImageBatchAnalysisResponse`):
  ```json
  {
    "batch_id": "string",
    "total_images": "integer",
    "total_bottles": "integer",
    "matched_bottles": "integer",
    "unmatched_bottles": "integer",
    "total_estimated_weight_grams": "float",
    "bottles": [ ... ],
    "clusters": [ ... ],
    "per_image_results": [ ... ]
  }
  ```

#### `POST /api/images/create-label`
Create a new label from bottles identified in an image analysis.
- **Request Body**:
  ```json
  {
    "image_id": "string",
    "bottle_indices": [integer, ...],
    "label": { ...LabelCreate object... }
  }
  ```
- **Response**: Details of the created label.

#### `POST /api/images/add-to-label`
Add selected bottles from an image analysis to an existing label (updates reference embedding).
- **Request Body**:
  - `image_id`: String
  - `bottle_indices`: List[integer]
  - `label_id`: Integer
  - `use_clip`: Boolean (optional)
- **Response**: Status message and updated sample count.

#### `GET /api/images/{image_id}/bottles`
Get all cropped bottle images from a previous analysis.
- **Path Parameters**: `image_id` (string)
- **Response**: List of bottle crops.

---

### 5. Bottle Management (Individual)

#### `POST /api/bottles/confirm`
Confirm labels for a list of bottles (used for operator approval).
- **Request Body**:
  ```json
  {
    "bottle_ids": [integer, ...],
    "label_id": "integer"
  }
  ```
- **Response**: Confirmation message.

#### `POST /api/bottles/new-label`
Create a new label from a list of bottles (e.g., from a cluster).
- **Request Body**:
  ```json
  {
    "bottle_ids": [integer, ...],
    "label": { ...LabelCreate object... }
  }
  ```
- **Response**: Details of the created label.

#### `POST /api/bottles/{bottle_id}/assign-label`
Assign a label to a specific bottle.
- **Path Parameters**: `bottle_id` (integer)
- **Query Parameters**:
  - `label_id`: Integer
- **Response**: Confirmation message.

#### `POST /api/bottles/{bottle_id}/reassign`
Reassign a bottle to a different label (manual correction).
- **Path Parameters**: `bottle_id` (integer)
- **Request Body**:
  ```json
  {
    "label_id": "integer",
    "source": "string (default: 'manual')"
  }
  ```
- **Response**: Confirmation message.

#### `POST /api/bottles/batch-reassign`
Reassign multiple bottles to a label at once.
- **Request Body**:
  ```json
  {
    "bottle_ids": [integer, ...],
    "label_id": "integer",
    "source": "string (default: 'manual')"
  }
  ```
- **Response**: Confirmation message.

#### `GET /api/bottles/{bottle_id}/similarities`
Get similarity scores between a bottle and all known labels.
- **Path Parameters**: `bottle_id` (integer)
- **Response**: List of `SimilarityScore` objects.
  ```json
  [
    {
      "label_id": "integer",
      "label_name": "string",
      "similarity": "float"
    },
    ...
  ]
  ```

---

### 6. System Configuration

#### `GET /api/config`
Get current runtime configuration.
- **Response** (`ConfigResponse`):
  ```json
  {
    "yolo_confidence": "float",
    "similarity_threshold": "float",
    "high_confidence_threshold": "float",
    "medium_confidence_threshold": "float",
    "vit_model": "string",
    "use_clip": "boolean",
    "clip_model": "string",
    "hdbscan_min_cluster_size": "integer",
    "hdbscan_min_samples": "integer"
  }
  ```

#### `PUT /api/config`
Update runtime configuration. Any field can be optional.
- **Request Body**: JSON object matching `ConfigResponse` (all fields optional).
- **Response**: Updated configuration.
