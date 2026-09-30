# Building and Learning a Bayesian Network

Compact implementation of the `llm_bn.ipynb` hands-on material.

It covers:
1. the Cloudy -> Rain/Sprinkler -> WetGrass Bayesian network;
2. explicit CPT construction and model validation;
3. exact inference for `P(Rain=1 | WetGrass=1)`;
4. independent enumeration as a correctness oracle;
5. synthetic-data generation;
6. maximum-likelihood parameter estimation;
7. Bayesian parameter estimation for sparse data;
8. an optional local-LLM prompt plus validation workflow.

Install once:

```bash
pip install -r requirements.txt
```

Then open `llm_bn.ipynb` and run top-to-bottom. The local coding-LLM cell is optional because it downloads model weights.
