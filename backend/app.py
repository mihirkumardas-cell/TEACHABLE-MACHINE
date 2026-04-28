"""
Learnix API Backend — Clean deployment build for Render / cloud hosting.
Extracted from Teachable Machine.py (API routes only, no embedded HTML).
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import io
import joblib
import uuid
import inspect
import sys
from typing import Optional, List, Any, Dict
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

app = FastAPI(title="Learnix API")

# Allow the Netlify frontend (and any origin during development) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In‑memory session store (per‑process; resets on redeploy)
# ---------------------------------------------------------------------------
STORE: Dict[str, Any] = {
    "last_dataset": None,
    "last_target": None,
    "last_task": None,
    "last_modelId": None,
}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]
    modelId: Optional[str] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_task(y: pd.Series) -> str:
    if y.dtype.name in ["object", "category"]:
        return "classification"
    try:
        unique = y.nunique()
    except Exception:
        unique = len(y.unique())
    return "classification" if unique <= 20 else "regression"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(**ohe_kwargs))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", cat_transformer, cat_cols)
        ],
        remainder="drop"
    )


def choose_model(task: str, incremental: bool = False):
    if task == "classification":
        return SGDClassifier(max_iter=1000, tol=1e-3) if incremental else RandomForestClassifier(n_estimators=100)
    else:
        return SGDRegressor(max_iter=1000, tol=1e-3) if incremental else RandomForestRegressor(n_estimators=100)


def save_artifact(model_pipeline: Pipeline, metadata: dict) -> str:
    modelId = str(uuid.uuid4())
    path = os.path.join(ARTIFACTS_DIR, f"modelId_{modelId}.joblib")
    joblib.dump({"pipeline": model_pipeline, "metadata": metadata}, path)
    return modelId


def _artifact_path_from_id(modelId: str) -> str:
    new_path = os.path.join(ARTIFACTS_DIR, f"modelId_{modelId}.joblib")
    if os.path.exists(new_path):
        return new_path
    legacy_path = os.path.join(ARTIFACTS_DIR, f"model_{modelId}.joblib")
    return legacy_path


def load_artifact(modelId: str) -> dict:
    path = _artifact_path_from_id(modelId)
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found")
    return joblib.load(path)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "app": "Learnix API",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/health", "/upload_csv", "/train", "/predict", "/models"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), target: str = Form(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in CSV")
    STORE["last_dataset"] = df
    STORE["last_target"] = target
    STORE["last_task"] = infer_task(df[target])
    return {"status": "ok", "rows": len(df), "target": target, "inferred_task": STORE["last_task"]}


@app.post("/train")
def train(train_size: float = 0.8, random_state: int = 42, incremental: bool = False):
    df = STORE.get("last_dataset")
    target = STORE.get("last_target")
    if df is None or target is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Use /upload_csv first.")

    X = df.drop(columns=[target])
    y = df[target]

    task = infer_task(y)
    STORE["last_task"] = task

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=random_state)

    preprocessor = build_preprocessor(X_train)
    model = choose_model(task, incremental=incremental)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    if incremental and hasattr(model, "partial_fit"):
        if task == "classification":
            classes = y_train.unique()
            chunk_size = max(1, int(len(X_train) / 5))
            for i in range(0, len(X_train), chunk_size):
                X_chunk = X_train.iloc[i:i+chunk_size]
                y_chunk = y_train.iloc[i:i+chunk_size]
                X_proc = preprocessor.fit_transform(X_chunk)
                model.partial_fit(X_proc, y_chunk, classes=classes)
        else:
            X_proc = preprocessor.fit_transform(X_train)
            model.partial_fit(X_proc, y_train)
    else:
        pipeline.fit(X_train, y_train)

    if incremental and hasattr(model, "predict"):
        y_pred = model.predict(preprocessor.transform(X_val))
    else:
        y_pred = pipeline.predict(X_val)

    metric = accuracy_score(y_val, y_pred) if task == "classification" else mean_squared_error(y_val, y_pred, squared=False)
    metadata = {
        "task": task,
        "metric": float(metric),
        "metric_name": "accuracy" if task == "classification" else "rmse",
        "n_rows": len(df),
    }

    if incremental:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    modelId = save_artifact(pipeline, metadata)
    STORE["last_modelId"] = modelId

    return {"status": "trained", "modelId": modelId, "metadata": metadata}


@app.post("/predict")
def predict(req: PredictRequest):
    modelId = req.modelId or STORE.get("last_modelId")
    if modelId is None:
        raise HTTPException(status_code=400, detail="No model specified and no trained model available.")

    try:
        data = load_artifact(modelId)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    pipeline_obj = data.get("pipeline")
    rows_df = pd.DataFrame(req.rows)

    try:
        preds = pipeline_obj.predict(rows_df)
    except Exception:
        try:
            pre = pipeline_obj.named_steps["preprocessor"]
            mdl = pipeline_obj.named_steps["model"]
            preds = mdl.predict(pre.transform(rows_df))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {"predictions": preds.tolist(), "modelId": modelId}


@app.get("/models")
def list_models():
    models = []
    for fname in os.listdir(ARTIFACTS_DIR):
        if fname.endswith(".joblib"):
            metadata = joblib.load(os.path.join(ARTIFACTS_DIR, fname)).get("metadata", {})
            if fname.startswith("modelId_"):
                modelId = fname.split("modelId_")[-1].split(".joblib")[0]
            else:
                modelId = fname.split("model_")[-1].split(".joblib")[0]
            models.append({"modelId": modelId, "metadata": metadata})
    return {"models": models}


# ---------------------------------------------------------------------------
# Entry point (local dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
