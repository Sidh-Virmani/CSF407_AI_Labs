# 8_BN — Bayesian Networks and Autoregressive Language Models

Minimal Python implementation for the Bayesian Networks lab.

## Files

- `dataset.py` — lab dataset.
- `first_order_model.py` — estimates `P(X_t | X_{t-1})`, predicts, samples, generates text, and checks normalisation.
- `second_order_model.py` — estimates `P(X_t | X_{t-2}, X_{t-1})`.
- `compare_models.py` — compares parameter count, zero-probability contexts, generation diversity, and sample coherence.
- `run_all.py` — runs all programs and saves output files.
- `answers.md` — concise answers to Questions 1–14 and LLM reflection.

## Run

```bash
python run_all.py
```

No external libraries are required.

The scripts generate:

- `first_order_output.txt`
- `second_order_output.txt`
- `comparison_output.txt`
