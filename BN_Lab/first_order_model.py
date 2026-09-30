from collections import defaultdict, Counter
import random
from dataset import DATASET, SELECTED_CONTEXTS

START = "<START>"
END = "<END>"


class FirstOrderLanguageModel:
    """First-order autoregressive model estimating P(X_t | X_{t-1})."""

    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.probabilities = {}

    @staticmethod
    def tokenize(sentence):
        return [START] + sentence.lower().split() + [END]

    def train(self, sentences):
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            for current, nxt in zip(tokens, tokens[1:]):
                self.transition_counts[current][nxt] += 1

        self.probabilities = {}
        for current, counts in self.transition_counts.items():
            total = sum(counts.values())
            self.probabilities[current] = {
                nxt: count / total for nxt, count in counts.items()
            }

    def distribution(self, current):
        return self.probabilities.get(current, {})

    def predict_next(self, current):
        dist = self.distribution(current)
        if not dist:
            return None
        return max(dist, key=dist.get)

    def sample_next(self, current):
        dist = self.distribution(current)
        if not dist:
            return None
        words = list(dist.keys())
        weights = list(dist.values())
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sampling", max_tokens=30):
        current = START
        output = []

        for _ in range(max_tokens):
            if mode == "greedy":
                nxt = self.predict_next(current)
            elif mode == "sampling":
                nxt = self.sample_next(current)
            else:
                raise ValueError("mode must be 'greedy' or 'sampling'")

            if nxt is None or nxt == END:
                break

            output.append(nxt)
            current = nxt

        return " ".join(output)

    def normalization_test(self, tolerance=1e-9):
        results = {}
        for current, dist in self.probabilities.items():
            total = sum(dist.values())
            results[current] = (total, abs(total - 1.0) <= tolerance)
        return results


def main():
    random.seed(42)

    model = FirstOrderLanguageModel()
    model.train(DATASET)

    print("FIRST-ORDER CONDITIONAL PROBABILITY TABLES")
    for word in SELECTED_CONTEXTS:
        print(f"P(next | {word}) = {model.distribution(word)}")
        print(f"Most probable next token: {model.predict_next(word)}")
        print()

    print("NORMALISATION TEST")
    all_ok = True
    for word, (total, ok) in model.normalization_test().items():
        all_ok &= ok
        print(f"{word:>8}: total={total:.6f} {'PASS' if ok else 'FAIL'}")
    print("Overall:", "PASS" if all_ok else "FAIL")

    print("\n20 SAMPLED SENTENCES")
    for i in range(20):
        print(f"{i + 1:02d}. {model.generate('sampling')}")

    print("\n5 GREEDY SENTENCES")
    for i in range(5):
        print(f"{i + 1:02d}. {model.generate('greedy')}")

    print("\n5 SAMPLED SENTENCES")
    for i in range(5):
        print(f"{i + 1:02d}. {model.generate('sampling')}")


if __name__ == "__main__":
    main()
