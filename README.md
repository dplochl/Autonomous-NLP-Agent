# Disaster Tweets Autonomous Research Agent

## Overview

This project implements an autonomous machine learning agent for the Kaggle competition:
https://www.kaggle.com/competitions/nlp-getting-started

The system:
- proposes experiments
- trains models
- evaluates performance
- logs results
- iterates automatically

---

## Project Structure

apa-disaster-tweets-agent/
│
├── data/        (not included)
├── logs/
├── outputs/
├── models/
│
├── src/
│   ├── config.py
│   ├── agent_vs.py
│   ├── agents/
│   │   ├── v1_simple.py
│   │   └── v2_transformer.py
│   ├── dashboard.py
│   └── test_ollama.py
│
├── requirements.txt
├── .gitignore
└── README.md

---

## Setup

Clone:

git clone git@github.com:Nic000111/apa-disaster-tweets-agent.git
cd apa-disaster-tweets-agent

Create env:

python3 -m venv .venv
source .venv/bin/activate

Install:

pip install -r requirements.txt

---

## Data

Download Kaggle data:

mkdir -p data
kaggle competitions download -c nlp-getting-started -p data
unzip data/nlp-getting-started.zip -d data

---

## Run

Agent runner (recommended):

python src/agent_vs.py v1

Transformer agent:

python src/agent_vs.py v2

Dashboard:

streamlit run src/dashboard.py

---

## Config

Optional:

export OLLAMA_MODEL="gemma4:e4b"
export HF_DEFAULT_MODEL="distilroberta-base"
export DISASTER_AGENT_DATA_DIR="/your/path/to/data"

---

## Results

Simple model ≈ 0.76 F1  
Transformer ≈ 0.83 F1  

---

## Notes

- data is not included
- models/logs not committed
- works with any local LLM via Ollama

---

## Next Steps

- cross validation
- ensemble models
- submission pipeline

---

## Agent_V2 (Autonomous ML Research Agent)

A new, fully autonomous agent is available in `src/Agent_V2/`. This agent uses a locally-hosted LLM (via Ollama) to drive the experimentation loop for the Kaggle Disaster Tweets competition.

### Architecture Overview

(Excerpt from `src/Agent_V2/ARCHITECTURE.md`)

> An agent that autonomously designs, trains, evaluates, and iterates on ML models for the [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) Kaggle competition. The agent uses a locally-hosted LLM (via Ollama) as its "brain" to drive the experimentation loop.

#### File Structure

```
src/Agent_V2/
├── agent_fully_autonomous.py  # Entry point — fully autonomous loop
├── llm.py            # Ollama LLM client (THINK + REFLECT steps)
├── memory.py         # Tiered experiment memory (rolling log + milestones)
├── sandbox.py        # Safe code execution (dry run + full run + monitor)
├── templates.py      # Boilerplate-correct code templates per architecture
├── prompts.py        # All LLM prompt templates
│
├── train.csv         # Kaggle training data (7613 tweets)
├── test.csv          # Kaggle test data (3263 tweets)
├── experiment_log.json  # Persistent log of all experiments (auto-generated)
├── submissions/      # Kaggle submission CSVs (auto-generated)
│
├── disaster_tweets.ipynb  # Manual baseline notebook
└── Untitled-1.py          # Original manual pipeline script
```

#### Agent Loop

```
AGENT LOOP (agent_fully_autonomous.py):
- MEMORY (project brief, rolling log, milestones)
- THINK (LLM proposes architecture + code)
- DRY RUN (1 epoch, 200 rows)
- EXECUTE (full run)
- REFLECT (LLM analyzes results)
- UPDATE MEMORY
- Repeat until F1 >= 0.88, plateau, or max iterations
```

#### Architecture Sequence

The agent explores architectures in this order:
1. BoW (TF-IDF + Logistic Regression)
2. BoW_advanced (multi-vectorizer ensemble)
3. BoW_advanced_thr (threshold tuning)
4. CNN (PyTorch)
5. LSTM (PyTorch)
6. Transformer (HuggingFace)

See `src/Agent_V2/ARCHITECTURE.md` for full details and design decisions.
