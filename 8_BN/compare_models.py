from first_order_model import FirstOrderLanguageModel
from second_order_model import SecondOrderLanguageModel
from dataset import DATASET

START = "<START>"


def zero_probability_contexts_first(model):
    # Among vocabulary tokens, count tokens with no outgoing observed transition.
    vocab = set()
    for sentence in DATASET:
        vocab.update(sentence.lower().split())
    vocab.add("<END>")
    return sum(1 for token in vocab if token not in model.probabilities)


def zero_probability_contexts_second(model):
    # Count possible vocabulary pairs that were never observed as a context.
    vocab = set()
    for sentence in DATASET:
        vocab.update(sentence.lower().split())
    vocab.add("<START>")
    vocab.add("<END>")
    possible = len(vocab) ** 2
    observed = len(model.probabilities)
    return possible - observed


def distinct_parameters(model):
    # Number of explicitly stored non-zero conditional probabilities.
    return sum(len(dist) for dist in model.probabilities.values())


def diversity(model, n=100):
    samples = [model.generate("sampling") for _ in range(n)]
    return len(set(samples)), n


def main():
    first = FirstOrderLanguageModel()
    second = SecondOrderLanguageModel()
    first.train(DATASET)
    second.train(DATASET)

    first_unique, n = diversity(first)
    second_unique, _ = diversity(second)

    print("MODEL COMPARISON")
    print("----------------")
    print(f"First-order distinct parameters:  {distinct_parameters(first)}")
    print(f"Second-order distinct parameters: {distinct_parameters(second)}")
    print(f"First-order zero-probability contexts:  {zero_probability_contexts_first(first)}")
    print(f"Second-order zero-probability contexts: {zero_probability_contexts_second(second)}")
    print(f"First-order generation diversity:  {first_unique}/{n} unique samples")
    print(f"Second-order generation diversity: {second_unique}/{n} unique samples")

    print("\nQUALITATIVE EXAMPLES")
    print("First-order:")
    for _ in range(5):
        print(" -", first.generate("sampling"))

    print("Second-order:")
    for _ in range(5):
        print(" -", second.generate("sampling"))


if __name__ == "__main__":
    main()
