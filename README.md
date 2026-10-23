# GroundPulse interactive dashboard

The project supports both local and static Netlify deployments.

## Run locally

On Windows, run `run.bat` and keep its terminal open. Open http://127.0.0.1:8765 in a browser. The local Python service loads the supplied feature CSV, trains the Decision Tree and Random Forest, reports a session-aware holdout evaluation, and classifies simulated windows.

## Deploy to Netlify

The site needs no Python runtime at request time on Netlify. `models.json` contains browser-usable tree structures exported from the supplied dataset by `export_models.py`; the frontend uses those trained models for predictions and graphs when the Python API is unavailable. Deploy the project root as the publish directory with no build command. The included `netlify.toml` sets the publish directory to `.`.

To refresh the trained browser model after changing the dataset or model settings, run `python export_models.py` in the local virtual environment, then commit and redeploy `models.json` with the site files.

## Project files

- `index.html`, `style.css`, `app.js`: dashboard, signal simulation, SP processing, graphs and browser model inference.
- `server.py`: local model training, evaluation and prediction API.
- `models.json` and `export_models.py`: static export of the trained Decision Tree and Random Forest for Netlify.
- `ml/dataset.csv`: supplied labeled synthetic dataset.
- `ml/ml_realistic_eval.py` and `ml/generate_dataset.py`: supplied ML scripts.
- `sp/notebooks`: notebooks extracted from the supplied Ideathon ZIP; the ZIP is retained at the project root.
- `requirements.txt`, `run.bat`: local Python setup and launch.

The browser simulates 2 kHz one-second frames. Its AECD approximation uses the updated reference (3.3 V, 1.644 V bias, approximately 10x first-stage gain). The signal path calculates the current six-feature FIR pipeline, then uses the trained models. Truck maps to Vehicle because the supplied labels do not include a separate truck class. No live hardware is connected.

The SP notebook and current ML CSV describe different filter/features (3–150 Hz/four features versus 5–400 Hz/six features). Align these before hardware inference.
