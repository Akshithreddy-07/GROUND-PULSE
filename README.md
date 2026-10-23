# GroundPulse interactive dashboard

This is a local full-stack project. The browser page is the interface; the Python service loads the supplied feature CSV, trains the submitted Decision Tree and Random Forest, reports a session-aware holdout evaluation, and classifies live simulated windows.

## Run it

On Windows, double-click run.bat. It creates a virtual environment, installs requirements.txt, and starts the service. Open http://127.0.0.1:8765 in a browser and keep the terminal open.

## Project files

- index.html, style.css, app.js: local interface and AECD/SP simulation.
- server.py: model training, evaluation and prediction API.
- ml/dataset.csv: supplied labeled synthetic dataset.
- ml/ml_realistic_eval.py and ml/generate_dataset.py: supplied ML scripts.
- sp/notebooks: notebooks extracted from the supplied Ideathon ZIP; the ZIP is retained at the project root.
- requirements.txt: Python dependencies.

The browser generates 2 kHz one-second synthetic frames. Its AECD approximation uses the updated reference (3.3 V, 1.644 V bias, approximately 10x first-stage gain). The signal path calculates the current six-feature FIR pipeline, then calls the trained Python models. Truck is a Vehicle subtype because the supplied labels do not include a separate truck class. No live hardware is connected.

The SP notebook and current ML CSV describe different filter/features (3–150 Hz/four features versus 5–400 Hz/six features). This must be resolved before hardware inference.
