from collections import defaultdict, Counter
import random
from dataset import DATASET

START = "<START>"
END = "<END>"


class SecondOrderLanguageModel:
    """Second-order autoregressive model estimating P(X_t | X_{t-2}, X_{t-1})."""

    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.probabilities = {}

    @staticmethod
    def tokenize(sentence):
        # Two START tokens give the model a valid 2-token context at generation time.
        return [START, START] + sentence.lower().split() + [END]

    def train(self, sentences):
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            for i in range(2, len(tokens)):
                context = (tokens[i - 2], tokens[i - 1])
                nxt = tokens[i]
                self.transition_counts[context][nxt] += 1

        self.probabilities = {}
        for context, counts in self.transition_counts.items():
            total = sum(counts.values())
            self.probabilities[context] = {
                nxt: count / total for nxt, count in counts.items()
            }

    def distribution(self, previous_two):
        return self.probabilities.get(tuple(previous_two), {})

    def predict_next(self, previous_two):
        dist = self.distribution(previous_two)
        if not dist:
            return None
        return max(dist, key=dist.get)

    def sample_next(self, previous_two):
        dist = self.distribution(previous_two)
        if not dist:
            return None
        words = list(dist.keys())
        weights = list(dist.values())
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sampling", max_tokens=30):
        context = [START, START]
        output = []

        for _ in range(max_tokens):
            if mode == "greedy":
                nxt = self.predict_next(context)
            elif mode == "sampling":
                nxt = self.sample_next(context)
            else:
                raise ValueError("mode must be 'greedy' or 'sampling'")

            if nxt is None or nxt == END:
                break

            output.append(nxt)
            context = [context[-1], nxt]

        return " ".join(output)

    def normalization_test(self, tolerance=1e-9):
        results = {}
        for context, dist in self.probabilities.items():
            total = sum(dist.values())
            results[context] = (total, abs(total - 1.0) <= tolerance)
        return results


def main():
    random.seed(42)

    model = SecondOrderLanguageModel()
    model.train(DATASET)

    print("SECOND-ORDER EXAMPLE CONDITIONAL PROBABILITY TABLES")
    examples = [
        ("<START>", "<START>"),
        ("<START>", "the"),
        ("the", "cat"),
        ("the", "dog"),
        ("cat", "sat"),
        ("dog", "ran"),
    ]
    for context in examples:
        print(f"P(next | {context}) = {model.distribution(context)}")

    print("\nNORMALISATION TEST")
    all_ok = True
    for context, (total, ok) in model.normalization_test().items():
        all_ok &= ok
        print(f"{context}: total={total:.6f} {'PASS' if ok else 'FAIL'}")
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
