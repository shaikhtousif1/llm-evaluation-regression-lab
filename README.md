# LLM Evaluation & Regression Lab

This is the next stage of the original local RAG banking assistant learning journey. The earlier `llmops-learning-lab` remains frozen as the baseline; this repository now brings its RAG code forward so evaluation tests the same approach rather than a separately reimplemented RAG pipeline.

## What this stage adds

- `src/rag_app.py`: the original local RAG flow (character chunks, Ollama embeddings, NumPy cosine similarity, top-K retrieval, Ollama generation).
- `src/evaluate.py`: calls the shared functions from `rag_app.py` against a fixed test set.
- `src/compare.py`: compares two saved evaluation reports.
- `data/banking_policy.md`: the same synthetic policy used in the original baseline.
- `data/test_cases.json`: six fixed questions and expected outcomes.
- `reports/`: generated JSON experiment outputs, ignored by Git.

The key continuity rule: **evaluation imports the same chunking, embedding, retrieval, and generation code as the interactive app**. There is no second RAG implementation to learn or maintain.

## Requirements and setup

- Python 3.10+
- Ollama running locally
- Models `llama3.2:1b` and `all-minilm`

    ollama pull llama3.2:1b
    ollama pull all-minilm

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run the interactive app

    python src/rag_app.py

Type `exit` to stop.

## Run the evaluation experiment

First evaluate the current top-K=3 baseline:

    python src/evaluate.py --top-k 3 --output reports/top3.json

Then evaluate top-K=1:

    python src/evaluate.py --top-k 1 --output reports/top1.json

Compare them:

    python src/compare.py reports/top3.json reports/top1.json

## Understand the metrics

- **Retrieval hit:** whether the expected policy heading appears in any retrieved chunk.
- **Answer fact:** whether each required fact group has at least one accepted literal phrase in the answer.
- **Abstention:** whether an unanswerable question gets an answer indicating the policy does not specify it.
- **Regression:** a check that passed in the baseline but fails in the candidate.

These are diagnostic checks, not proof of semantic correctness. A phrase check may fail on a correctly worded answer or pass on a misleading answer. Inspect the retrieved chunk text, model answer, and original policy.

## Learning approach

We are adding one concept to the existing RAG knowledge: systematic evaluation. First understand the output and its failure modes; then make one improvement and measure its effect. Do not add vector databases, agents, Kubernetes, dashboards, or hosted APIs yet.
