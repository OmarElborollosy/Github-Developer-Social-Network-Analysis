# 🚀 GitHub Social Network Analysis & Interactive Dashboard

This repository contains a comprehensive Social Network Analysis (SNA) of the GitHub developer collaboration network (MUSAE dataset). The project explores community structures, developer influence, and information diffusion dynamics between Web and Machine Learning (ML) developers.

---

## 📊 Project Highlights
- **Analytical Rigor:** Validated the Small-World phenomenon with a sigma value of **935.55**.
- **Community Detection:** Identified 31 distinct communities using the Louvain algorithm (Modularity Q = 0.4534).
- **Influence Modeling:** Simulated Independent Cascade (IC) and Linear Threshold (LT) models to map information spread.
- **Link Prediction:** Achieved **0.8474 AUC** using the Resource Allocation index with strict zero-leakage protocols.

---

## 🛠️ Setup Instructions

### 1. Environment Preparation
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

```powershell
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries using the provided `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

## 🖥️ Launching the Dashboard

The interactive Streamlit dashboard allows you to visualize the network, explore centrality rankings, and run diffusion simulations in real-time.

To launch the dashboard from the project root, run:

```powershell
streamlit run dashboard/app.py
```

> **Note:** On first launch, the dashboard will load the pre-computed LCC graph from `outputs/graph_lcc.pkl` for maximum performance.

---

## 📓 Running the Analysis Notebook

The full analytical pipeline, including data cleaning, metric calculation, and figure generation, is contained in the Jupyter Notebook:

```powershell
jupyter notebook notebooks/analysis.ipynb
```

---

## 📂 Project Structure
- `dashboard/`: Contains the Streamlit app and utility functions.
- `notebooks/`: The main `analysis.ipynb` containing the full research pipeline.
* `report/`: The final research paper (`GitHub_SNA_Paper_v2.md`).
- `data/`: Raw dataset files (`musae_git_edges.csv`, `musae_git_target.csv`).
- `outputs/`: 
    - `figures/`: Generated plots and visualizations.
    - `results/`: CSV files containing calculated centrality and community metrics.

---

## 📜 Dataset Reference
**SNAP MUSAE GitHub Dataset:** 
Rozemberczki, B., Allen, C., & Sarkar, R. (2021). *Multi-Scale Attributed Node Embedding*. Journal of Complex Networks.
Available at: [https://snap.stanford.edu/data/git_web_ml.zip](https://snap.stanford.edu/data/git_web_ml.zip)
