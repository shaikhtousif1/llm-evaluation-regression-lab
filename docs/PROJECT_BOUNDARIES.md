# Project Boundaries

## Learning objective

Learn to evaluate an LLM/RAG application and detect regressions when a configuration changes.

## The first experiment

Run the same fixed questions with `top_k=3` and `top_k=1`. Compare:

- Did the expected policy section appear in retrieval?
- Did answerable questions include required facts?
- Did unanswerable questions appropriately abstain?
- Which individual cases regressed?

## What this project is not

It is not a production RAG platform. Do not add a vector database, agent framework, web UI, Kubernetes, observability stack, or hosted model integration to the first version.

## Known limitations

- The test set is small and hand-authored.
- Required-fact checks use string alternatives; they can miss correct paraphrases and accept some incorrect statements.
- Abstention checks search for a small set of phrases.
- A metric is a signal for investigation, not proof of quality.
- Local model outputs can vary across model versions and runtime settings.
- A production evaluation would need broader cases, human review, and stronger semantic/groundedness checks.
