import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.manual_seed(17)


def translation_demo():
    model_id = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(DEVICE).eval()

    examples = [
        "Hello, how are you today?",
        "Artificial intelligence can help us solve difficult problems.",
        "The weather is beautiful, so we are going for a walk.",
    ]

    encoded = tokenizer(
        examples,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)

    generated = model.generate(
        **encoded,
        max_new_tokens=60,
        num_beams=4,
        early_stopping=True,
    )
    translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

    assert len(translations) == len(examples)
    assert all(text.strip() for text in translations)

    print("=== Encoder-decoder: English -> French ===")
    for english, french in zip(examples, translations):
        print("English:", english)
        print("French :", french)
        print()

    del model, tokenizer
    gc.collect()


def causal_lm_demo():
    model_id = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE).eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quick brown fox jumps over the"
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    logits = model(**encoded).logits[0, -1]
    probabilities = torch.softmax(logits, dim=-1)
    values, ids = torch.topk(probabilities, k=5)

    print("=== Decoder-only: next-token prediction ===")
    print("Prompt:", prompt)
    for rank, (prob, token_id) in enumerate(zip(values, ids), start=1):
        token = tokenizer.decode([int(token_id)])
        print(f"{rank}. {token!r}  probability={float(prob):.4f}")

    completion_ids = model.generate(
        **encoded,
        max_new_tokens=25,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
    assert completion.startswith(prompt)
    assert len(completion) > len(prompt)

    print("\nCompletion:")
    print(completion)
    print()

    del model, tokenizer
    gc.collect()


def sentiment_demo():
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval()

    examples = [
        "This movie was fantastic and I enjoyed every minute.",
        "The service was slow, rude, and extremely disappointing.",
        "The new update is excellent and much easier to use.",
        "I regret buying this product because it stopped working immediately.",
    ]
    expected = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]

    encoded = tokenizer(
        examples,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)
    probabilities = torch.softmax(model(**encoded).logits, dim=1)

    predictions = []
    print("=== Encoder classifier: sentiment ===")
    for text, vector in zip(examples, probabilities):
        class_id = int(vector.argmax().item())
        label = model.config.id2label[class_id]
        confidence = float(vector[class_id].item())
        predictions.append(label)

        assert abs(float(vector.sum().item()) - 1.0) < 1e-6
        print("Text:", text)
        print(f"Prediction: {label}  confidence={confidence:.4f}")
        print()

    assert predictions == expected

    del model, tokenizer
    gc.collect()


def main():
    print("Device:", DEVICE)
    translation_demo()
    causal_lm_demo()
    sentiment_demo()
    print("All Transformer task checks passed.")


if __name__ == "__main__":
    main()
