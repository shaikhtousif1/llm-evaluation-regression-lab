# LLM Evaluation & Regression Lab

A small, local-first project for learning one advanced LLMOps capability: **measuring whether a RAG application gets better or worse when its configuration changes**.

This repository is deliberately separate from `llmops-learning-lab`. It is not a production platform and it does not use an evaluation framework.

## What it does

1. Reads a small synthetic banking policy document.
2. Splits it into heading-aware sections.
3. Uses Ollama and `all-minilm` to embed sections and questions.
4. Retrieves the top-K sections using cosine similarity.
5. Uses `llama3.2:1b` to answer from retrieved context.
6. Checks whether the expected source section was retrieved.
7. Checks expected answer facts or appropriate abstention.
8. Writes a JSON report that can be compared with another run.

## Requirements

- Python 3.10+
- Ollama running locally
- Models downloaded:
  ```bash
  ollama pull llama3.2:1b
  ollama pull all-minilm
  ```

## Run locally

From the project root, in Ubuntu/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/evaluate.py --top-k 3 --output reports/top3.json
```

Then run a different retrieval configuration:

```bash
python src/evaluate.py --top-k 1 --output reports/top1.json
python src/compare.py reports/top3.json reports/top1.json
```

The model runs locally. No hosted API key is required. Evaluation calls the local models for each test case and may take a few minutes.

## How to read the results

- **Retrieval hit rate**: fraction of questions where the expected policy heading appears in retrieved results.
- **Answer fact pass rate**: fraction of answerable cases where all required fact groups appear in the answer. Each group contains accepted text alternatives.
- **Abstention pass rate**: fraction of unanswerable cases where the answer says the source does not specify the answer.
- **Regressions**: test cases that passed a metric in the baseline report but fail it in the candidate report.

These simple deterministic checks are imperfect proxies for semantic correctness. Wording differences can cause false failures, and expected words can appear in an incorrect answer. Inspect failed cases manually.

## Learning experiment

Run the same test set with `top_k=3` and `top_k=1`. Compare the reports. Did fewer chunks improve retrieval precision but remove needed evidence? Did answer quality change? The goal is to observe and explain the trade-off, not assume one setting is better.

## Scope boundaries

In scope:
- A small fixed evaluation dataset
- Local RAG execution
- Separate retrieval and answer checks
- JSON reports
- Baseline-versus-candidate comparison

Out of scope for this first version:
- Vector databases
- Agents or tool calling
- Kubernetes or deployment pipelines
- Hosted model APIs
- Dashboards and tracing platforms
- LLM-as-a-judge
- Large evaluation frameworks

## Safety note

The banking content is fictional and for training only. It is not financial advice.
