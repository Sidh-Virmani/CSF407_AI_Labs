# Transformer Models Lab

The notebook demonstrates three Transformer use cases:

1. **Encoder-decoder:** English -> French translation.
2. **Decoder-only:** next-token probabilities and text completion.
3. **Encoder + classification head:** sentiment classification.

Install:

```bash
pip install -r requirements.txt
```

Then run `transformers.ipynb`.

The first execution downloads pretrained model weights from Hugging Face, so internet access is required. CPU execution is sufficient, although downloads/inference can take a few minutes.
