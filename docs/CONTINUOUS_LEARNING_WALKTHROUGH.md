# Evaluation Stage — Building on the Original RAG App

## Why this repository exists

The earlier `llmops-learning-lab` demonstrated the RAG request flow and was intentionally stopped as a small, understandable baseline. This repository continues from that code. It does not ask you to learn a new RAG design from scratch.

## Shared architecture

The original RAG flow remains:

1. Read the banking policy.
2. Split it into overlapping character chunks (700 characters, 100 overlap).
3. Embed chunks with Ollama's `all-minilm`.
4. Embed a question and rank chunks using NumPy cosine similarity.
5. Select top-K chunks.
6. Pass their text, the question, and grounding instructions to `llama3.2:1b`.
7. Print the answer and retrieval scores.

`src/rag_app.py` contains these shared functions. The interactive CLI uses them, and `src/evaluate.py` imports and reuses them. This prevents the evaluator from quietly testing a different implementation.

## What is new

The original app was checked manually. Manual testing is useful for a first smoke test but becomes unreliable when configuration changes. We add a fixed set of questions in `data/test_cases.json` and an evaluation runner that saves the evidence and results to JSON.

For each test, the evaluator records:
- retrieved chunk IDs, scores, and text;
- whether the expected heading appeared in retrieved evidence;
- the generated answer;
- required-fact checks for answerable questions;
- abstention checks for unanswerable questions.

`src/compare.py` compares two reports and lists pass-to-fail changes.

## First experiment: top-K

Run the same questions with top-K=3 and top-K=1. The only intended configuration change is how many chunks are passed to the generation model.

- More chunks may include needed exceptions, but can add irrelevant context.
- Fewer chunks may reduce noise, but can omit a qualification or exception.
- Retrieval hit is not answer correctness: generation can still misunderstand or invent details.

The checks use literal phrases and are intentionally simple. Treat them as signals. Read the original policy, retrieved evidence, and answer before deciding what failed.

## Commands

    python src/rag_app.py
    python src/evaluate.py --top-k 3 --output reports/top3.json
    python src/evaluate.py --top-k 1 --output reports/top1.json
    python src/compare.py reports/top3.json reports/top1.json

## Out of scope for now

No hosted model API, vector database, agents, Kubernetes, deployment pipeline, dashboard, or complex evaluation framework. Add one concept only after the current experiment is understood.
