# Cats vs Dogs Classifier API

A small end-to-end machine learning project that classifies an input image as **cat** or **dog**.  
It includes: dataset preparation, exploratory data analysis (EDA), model training + tuning, and an API for inference (FastAPI), containerized with Docker and deployed to the cloud (Render).

---

## 1) Problem Description

**Problem:**  
Given a photo containing a cat or a dog, predict which class it belongs to.

**Why it matters / usage:**  
This is a common baseline computer vision classification task. The deployed API can be used by any client (browser, mobile app, or scripts) to upload an image and get a prediction result.

**Output:**  
The API returns:
- `class`: `"cats"`, `"dogs"`, or `"uncertain"` (if confidence < threshold)
- `prob`: the maximum probability
- `probs`: per-class probabilities

---

## 2) Project Structure

```text
.
├── app.py                  # FastAPI inference service
├── train.py                # training script (exported from notebook logic)
├── predict.py              # (optional) local prediction helper script
├── notebook.ipynb          # EDA + experiments
├── requirements.txt        # dependencies
├── Dockerfile              # containerization
├── .dockerignore
├── .gitignore
├── models/
│   ├── model.pt            # final model used for deployment
│   ├── meta.json           # model metadata (classes, arch, etc.)
│   └── results_summary.json
└── data/                   # ignored in git (downloaded/processed locally)
```
Notes:
- The dataset is **NOT committed** to git (see `.gitignore`).
- Deployment artifacts are kept in `models/` (`model.pt`, `meta.json`).

---

## 3) Dataset

This project uses the Kaggle "Cats and Dogs" dataset:
- Source: `kushlesh kumar/cats-and-dogs`
- Original structure: `train/` and `validation/` with `cats/` and `dogs/`

### Local data layout (example)

```text
data/
└── processed/cats_dogs_70_30/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    ├── validation/
    │   ├── cats/
    │   └── dogs/
    └── test/
        ├── cats/
        └── dogs/
```

---

## 4) EDA (Exploratory Data Analysis)

The EDA is in `notebook.ipynb`. Summary:

- **Class balance**: checked counts for cats/dogs across splits to ensure no severe imbalance.
- **Image integrity**: verified file formats and tested random samples with PIL `verify()` to ensure no corrupted images.
- **Image sizes / aspect ratio**:
  - width/height distributions are right-skewed (some very large outliers)
  - aspect ratios mostly around ~1.4–1.8 with some outliers
  - decided to use resize/crop to standard model input size (e.g., 224x224)
- **Brightness distribution**: brightness varies significantly → augmentation (brightness/contrast) is helpful.

---

## 5) Model Training

Training code is in:
- `notebook.ipynb` (experiments)
- `train.py` (script version for reproducibility)

### Models tried
- **Baseline**: pretrained `ResNet18`
- **Improved**: pretrained `MobileNetV3-small`

### Tuning / Experiments
- Learning-rate sweep for MobileNetV3 with weight decay
- Compared multiple LR values and evaluated with validation metrics

### Example results (from experiments)
- Baseline ResNet18 achieved good performance
- MobileNetV3-small performed better after tuning (higher validation/test accuracy)

Final deployment model is saved to:
- `models/model.pt`
- `models/meta.json`

---

## 6) Training Script (train.py)

The training logic is exported to `train.py`.

Example usage (adjust paths/args if needed):

```bash
python train.py
```
Expected behavior:
	•	trains model on data/processed/.../train
	•	evaluates on validation set
	•	saves best model to models/model.pt and metadata to models/meta.json

Note: training requires the dataset to be present locally (see Dataset section).

## 7) Model Deployment (FastAPI)

The inference service is implemented in app.py using FastAPI.

### Endpoints
	•	GET /health
Returns service status and basic model metadata.
	•	POST /predict
Upload an image file (multipart/form-data) and get prediction.

### Uncertainty threshold

If model confidence is low:
	•	if prob < 0.60 → class = "uncertain"

You can change the threshold in app.py if needed.

⸻

## 8) Dependency & Environment Management (2 pts)

### Create & activate venv (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Run locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open Swagger UI:
	•	http://localhost:8000/docs

## 9) Containerization (Docker) (2 pts)

### Build image
```bash
docker build -t cats-dogs-api:latest .
```
### Run container
```bash
docker run --rm -p 8000:8000 cats-dogs-api:latest
```
### Verify
	•	http://localhost:8000/docs
or
```bash
curl http://localhost:8000/health
```
## 10) Cloud Deployment (Render)

This service is deployed as a Docker web service on Render.

Public URL:
- https://cats-dogs-classifier-api.onrender.com

Swagger docs:
- https://cats-dogs-classifier-api.onrender.com/docs

Health check:
- https://cats-dogs-classifier-api.onrender.com/health

Predict endpoint:
- https://cats-dogs-classifier-api.onrender.com/predict

Note: Render free tier may spin down with inactivity; the first request can take ~30–60 seconds to wake up.

## Test with curl (example)

Replace /path/to/cat.jpg with your local file path:

```bash
curl -X POST "https://cats-dogs-classifier-api.onrender.com/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/cat.jpg;type=image/jpeg"
```

### Example Response
```json
{
  "class": "cats",
  "prob": 0.98,
  "probs": {
    "cats": 0.98,
    "dogs": 0.02
  }
}
```
### Notes
	•	Render free tier may sleep when inactive, so first request can be slower.
	•	Dataset is not committed; model artifacts are committed for reproducible deployment.

## 11) Reproducibility

This repo provides step-by-step instructions to rerun the notebook and the training script.

### Data availability

	•	The dataset is not committed to the repository (large files).
	•	Please download and prepare the dataset locally following Section 3) Dataset.
	•	Expected layout: data/processed/cats_dogs_70_30/{train,validation,test}/{cats,dogs} (see Section 3).

### Re-run the notebook (EDA + experiments)

Open notebook.ipynb and run all cells.
It contains dataset checks, EDA, and model experiments.

Run training from script (exported from notebook)

train.py contains the training logic exported from the notebook, so training can be reproduced without the notebook UI.

Example:
```bash
python train.py --data-dir data/processed/cats_dogs_70_30
```

If your script uses different arguments, run python train.py --help and update the command accordingly.

### Tracked deployment artifacts

For reproducible deployment (API/Render), model artifacts are committed under models/:
	•	models/model.pt (final model used by the API)
	•	models/meta.json (class names and metadata)
	•	models/results_summary.json (training/evaluation summary)




