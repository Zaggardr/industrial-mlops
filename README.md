# Industrial MLOps

Project structure for an industrial machine-learning workflow.

## Directories

- `data/raw`: source datasets
- `data/processed`: transformed datasets
- `notebooks`: exploratory analysis and experiments
- `src/data`: data loading and validation
- `src/features`: feature engineering
- `src/models`: training and inference code
- `src/api`: model-serving API
- `tests`: automated tests
- `mlruns`: local MLflow tracking data

## Setup

```bash
pip install -r requirements.txt
```

The Docker Compose service exposes the API on port 8000 once `src/api/main.py` is added.
