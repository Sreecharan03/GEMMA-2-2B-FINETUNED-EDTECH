# JoSAA AI Assistant + NIRF Model Workspace

This repository is a mixed workspace with:

1. A **Streamlit app** (`main.py`) that answers JoSAA counselling queries.
2. **NIRF model/data notebooks** used for scraping, fine-tuning prep, and GGUF conversion experiments.

If you are new, start with the app section below.

## What This Project Does

The app lets a student ask questions like:

- "I have AIR 6000, which IIT programs can I get?"
- "Electrical options near rank 8000"

Behind the scenes, it:

1. Rewrites your question into better query variants.
2. Generates safe SQL.
3. Runs SQL on a PostgreSQL database.
4. Returns a plain-language counselling summary.

## Quick Start (Beginners)

### 1) Requirements

- Python 3.10+
- Internet access (for Gemini API + remote DB)

### 2) Create environment and install packages

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit pandas psycopg2-binary google-generativeai
```

### 3) Check credentials in `main.py`

`main.py` currently contains:

- PostgreSQL connection config (`DB = {...}`)
- Gemini API key (`GEMINI_API_KEY = "..."`)

If you run your own setup, replace these with your own values.

### 4) Run the app

```bash
streamlit run main.py
```

Then open the local Streamlit URL shown in terminal (usually `http://localhost:8501`).

## Important Notes

- The chat app uses a **PostgreSQL table** (generated SQL expects JoSAA-style tables such as `josaa_btech_2024`).
- `josaa_data.db` exists in this repo, but `main.py` does **not** use SQLite by default.
- `requirements.txt` in this repo is notebook-oriented and does not include all app UI packages, so use the install command above for app usage.

## Repository Guide

- `main.py`: Main Streamlit JoSAA assistant (current primary app).
- `use_gguf_model.sh`: Example commands for running a GGUF model with `llama.cpp`.
- `web_scrap.ipynb`: Scrapes NIRF pages and creates cleaned datasets.
- `rag_1.ipynb`: JoSAA 2024 dataset analysis/transformation experiments.
- `rag_2.ipynb`, `rag_3.ipynb`: GGUF inference experiments for NIRF Q&A.
- `gemma_quantization.ipynb`, `gemma_quantizationv2_0.ipynb`: HF -> GGUF conversion and quantization attempts.
- `data/clean`: Clean CSV data (example: NIRF 2025 overall rankings).
- `data/sft`: JSONL files used for NIRF lookup fine-tuning style data.
- `converted_gguf`, `gguf_output`, `model`, `models`: Generated/downloaded model artifacts.
- `llama.cpp`: Local copy of llama.cpp used for conversion/inference tooling.

## Running GGUF Inference (Optional)

If you only want local model inference (not the JoSAA app):

1. Build `llama.cpp`.
2. Use `use_gguf_model.sh` as a reference command script.
3. Point to one of the GGUF files in `converted_gguf/` or `gguf_output/`.

## Troubleshooting

- `ModuleNotFoundError: streamlit` or `pandas`: run the pip install command from Quick Start.
- DB connection failure: verify host/user/password/port in `main.py`.
- Gemini error (401/403): verify `GEMINI_API_KEY`.
- Very slow first run: model/API and schema-loading steps can take time on first launch.

## Security Warning

Some notebooks and scripts contain hard-coded tokens/keys. Treat this repo as a private workspace and rotate credentials before sharing publicly.
