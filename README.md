# Heart Disease Prediction (Modular Python Project)

This project is a refactored version of your Colab notebook, organized into reusable Python modules with a clean entry point.

## Project Structure

```text
disease_predn/
├─ main.py
├─ requirements.txt
├─ .gitignore
├─ data/
├─ artifacts/
└─ src/
   └─ disease_prediction/
      ├─ __init__.py
      ├─ config.py
      ├─ data.py
      ├─ features.py
      ├─ models.py
      ├─ evaluation.py
      └─ pipeline.py
```

## What Was Refactored

- Notebook steps were split into focused modules.
- Repetitive/non-useful print statements were removed.
- Evaluation outputs are now saved to files instead of noisy console logs.
- A single `main.py` script orchestrates end-to-end training.

## Setup

1. Create and activate virtual environment:
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Data

Put your CSV at:

- `data/heart.csv`

Or pass any custom path using `--data-path`.

## Run Training

```powershell
python main.py --data-path data/heart.csv
```

Optional output folder:

```powershell
python main.py --data-path data/heart.csv --artifacts-dir artifacts
```

## Outputs

After running, these files are created in `artifacts/`:

- `model.pkl` - best tuned random forest model
- `metrics.json` - all model metrics and threshold analysis
- `feature_importance.csv` - random forest feature importance

## Next Step (GitHub)

When you are ready, initialize git and push:

```powershell
git init
git add .
git commit -m "Refactor notebook into modular Python project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

