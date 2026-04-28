# Learnix API

This project is a FastAPI app with a modern Learnix frontend that lets you:

- upload a CSV file
- choose a target column
- train a machine learning model
- save the trained model
- predict on new rows

## Files

- `Teachable Machine.py`: main FastAPI app
- `web/index.html`: Learnix UI
- `requirements.txt`: Python dependencies
- `sample_dataset.csv`: sample dataset for testing
- `artifacts/`: saved trained models
- `netlify.toml`: Netlify static deployment config for `web/`

## Setup

Create and activate a virtual environment if you want:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

Start the app with:

```powershell
python "Teachable Machine.py"
```

Or double-click:

- `start_teachable_machine.bat`

After starting, open:

- `http://127.0.0.1:8000/` (Learnix UI)
- `http://127.0.0.1:8000/docs` (API docs)

## How It Works

1. Upload a CSV file to `/upload_csv`
2. Pass the target column name
3. Call `/train` to train a model
4. Call `/predict` with JSON rows to get predictions
5. Call `/models` to see saved models

## Sample Test

Use `sample_dataset.csv` and set:

- `target = label`

Then send a prediction request like:

```json
{
  "rows": [
    { "age": 30, "city": "A", "income": 62000 }
  ]
}
```

## Endpoints

- `POST /upload_csv`
- `POST /train`
- `POST /predict`
- `GET /models`
- `GET /health`

## Netlify Deployment (Frontend)

1. Deploy this repo to Netlify (publish directory is already set to `web` in `netlify.toml`).
2. Keep your FastAPI backend running on a public URL (Render/Railway/Fly/VM, etc.).
3. In Learnix UI on Netlify, set **API base URL** to your backend URL.

Example API base URL:

```text
https://your-backend-domain.com
```
