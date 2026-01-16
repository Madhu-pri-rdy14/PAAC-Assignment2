
This project implements a **Transformer-based model** to reconstruct quantum density matrices from "Classical Shadow" measurement data. It was developed as part of the PAAC Open Project .

## 📂 Project Structure
* **`src/`**: Source code for the model, training, and data generation.
* **`outputs/`**: Contains the trained model weights (`model_track1.pt`).
* **`docs/`**: Documentation and detailed analysis.

## 📄 Documentation
Please see [docs/report.md](docs/report.md) for the full details, including:
* Mathematical Model (Cholesky Decomposition)
* Replication Guide (How to run the code)
* Final Results & Analysis

## 🤖 AI Attribution
Transparency regarding AI tools used in this project can be found in [AI_USAGE.md](AI_USAGE.md).

## 🚀 Quick Start
To reproduce the results:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python src/train.py

# 3. Measure latency
python src/inference.py
