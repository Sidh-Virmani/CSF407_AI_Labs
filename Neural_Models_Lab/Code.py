"""AI Hands-on: Neural Models - XOR, backpropagation, symmetry and output layers.

Run with:
    python Code.py

The script performs every required executable experiment from the lab:
1) a 2-2-1 XOR network with a nonlinear hidden layer,
2) a first-layer gradient/backpropagation check,
3) zero-initialisation symmetry experiment,
4) sigmoid/tanh/ReLU activation comparison,
5) a 2-2-3 three-class extension with softmax probabilities.
"""

import torch
import torch.nn as nn

# The dataset is tiny. One CPU thread keeps the run fast and deterministic enough
# for this laboratory exercise.
torch.set_num_threads(1)

X = torch.tensor(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]],
    dtype=torch.float32,
)

# Binary XOR targets: warning iff exactly one sensor is active.
Y_BINARY = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Three-class targets:
# 0 -> both inactive, 1 -> disagreement, 2 -> both active.
Y_MULTI = torch.tensor([0, 1, 1, 2], dtype=torch.long)

SEED = 2
STEPS = 3000
LEARNING_RATE = 0.05


def set_seed(seed=SEED):
    """Reset PyTorch's random generator so experiments are reproducible."""
    torch.manual_seed(seed)


class TinyNetwork(nn.Module):
    """2 hidden units with a selectable nonlinearity and configurable output size."""

    def __init__(self, activation="tanh", output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)              # 2 inputs -> 2 hidden units
        self.fc2 = nn.Linear(2, output_size)    # hidden units -> output logits
        self.activation = activation

    def hidden(self, x):
        """Compute hidden pre-activations and apply the requested nonlinearity."""
        z = self.fc1(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(z)
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "relu":
            return torch.relu(z)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        """Return logits. Sigmoid/softmax are used only when reporting probabilities."""
        return self.fc2(self.hidden(x))


def train_binary(activation="tanh", *, zero_init=False, steps=STEPS):
    """Train the required 2-2-1 XOR network with full-batch Adam optimisation."""
    set_seed()
    model = TinyNetwork(activation=activation, output_size=1)

    # For the symmetry experiment we initialise every trainable parameter identically.
    if zero_init:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

    loss_fn = nn.BCEWithLogitsLoss()  # stable sigmoid + binary cross-entropy pairing
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with torch.no_grad():
        initial_loss = loss_fn(model(X), Y_BINARY).item()

    early_gradient = None
    symmetry_snapshots = []
    snapshot_steps = {0, 1, 10, 100, 500}

    for step in range(steps):
        # Store the hidden-layer rows before selected updates in the zero-init run.
        if zero_init and step in snapshot_steps:
            symmetry_snapshots.append((step, model.fc1.weight.detach().clone()))

        optimizer.zero_grad()
        logits = model(X)                 # forward pass
        loss = loss_fn(logits, Y_BINARY)  # scalar mean loss over the four examples
        loss.backward()                   # reverse-mode AD / backpropagation

        # Capture a real gradient tensor immediately after backward() and before step().
        if step == 0:
            early_gradient = model.fc1.weight.grad.detach().clone()

        optimizer.step()                  # update trainable parameters

    if zero_init and steps in snapshot_steps:
        symmetry_snapshots.append((steps, model.fc1.weight.detach().clone()))

    # One final forward/backward pass exposes the trained model's first-layer gradient too.
    optimizer.zero_grad()
    final_logits = model(X)
    final_loss_tensor = loss_fn(final_logits, Y_BINARY)
    final_loss_tensor.backward()

    with torch.no_grad():
        probabilities = torch.sigmoid(final_logits).squeeze(1)
        predictions = (probabilities >= 0.5).to(torch.int64)

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss_tensor.item(),
        "probabilities": probabilities.detach().clone(),
        "predictions": predictions.detach().clone(),
        "early_gradient": early_gradient,
        "early_gradient_norm": early_gradient.norm().item(),
        "final_gradient": model.fc1.weight.grad.detach().clone(),
        "symmetry_snapshots": symmetry_snapshots,
    }


def print_binary_baseline():
    """Task 4A/4B: successful XOR learning plus a backpropagation gradient check."""
    print("\n=== A. Binary XOR learning (2 -> 2 -> 1, tanh hidden) ===")
    result = train_binary("tanh")

    print(f"Initial loss: {result['initial_loss']:.6f}")
    print(f"Final loss:   {result['final_loss']:.6f}")
    print("\nInput      Probability   Predicted   Target")
    targets = Y_BINARY.squeeze(1).to(torch.int64)
    for x, p, pred, target in zip(X, result["probabilities"], result["predictions"], targets):
        print(f"{tuple(int(v) for v in x.tolist())!s:<10} {p.item():.6f}      {pred.item()}           {target.item()}")

    print("\nFirst-layer gradient after the first backward():")
    print(result["early_gradient"])
    print(f"Early gradient norm: {result['early_gradient_norm']:.6f}")

    assert torch.equal(result["predictions"], targets), "Binary XOR did not learn all four labels."
    assert result["final_loss"] < result["initial_loss"], "Binary loss did not decrease."
    assert result["early_gradient_norm"] > 0.0, "Expected a nonzero learning signal."
    print("Checks: PASS - loss decreased, gradient was nonzero, and all 4 labels are correct.")


def print_symmetry_experiment():
    """Task 4C: show why identical zero initialisation fails to break hidden-unit symmetry."""
    print("\n=== B. Zero-initialisation symmetry experiment ===")
    result = train_binary("tanh", zero_init=True, steps=500)

    all_equal = True
    for step, weights in result["symmetry_snapshots"]:
        equal = torch.allclose(weights[0], weights[1])
        all_equal = all_equal and equal
        print(f"Step {step:>3}: row 1 = {weights[0].tolist()}, row 2 = {weights[1].tolist()}, identical = {equal}")

    print(f"Final loss with zero initialisation: {result['final_loss']:.6f}")
    print(f"Final probabilities: {[round(v, 6) for v in result['probabilities'].tolist()]}")

    assert all_equal, "The two hidden rows should remain identical in the symmetry experiment."
    print("Check: PASS - the hidden units remain identical, so they cannot learn distinct features.")


def print_activation_experiment():
    """Task 4D: change only the hidden activation and compare measured outcomes."""
    print("\n=== C. Hidden-activation experiment ===")
    print("Activation  Final loss    4/4 correct?  Early ||grad W1||_2")
    results = {}
    targets = Y_BINARY.squeeze(1).to(torch.int64)

    for activation in ("sigmoid", "tanh", "relu"):
        result = train_binary(activation)
        correct = bool(torch.equal(result["predictions"], targets))
        results[activation] = result
        print(f"{activation:<10}  {result['final_loss']:<12.6f}  {str(correct):<12}  {result['early_gradient_norm']:.6f}")

    return results


def train_three_class():
    """Task 5: keep the small hidden layer but change the output to three logits."""
    set_seed()
    model = TinyNetwork(activation="tanh", output_size=3)
    loss_fn = nn.CrossEntropyLoss()  # internally applies log-softmax + NLL loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with torch.no_grad():
        initial_loss = loss_fn(model(X), Y_MULTI).item()

    for _ in range(STEPS):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y_MULTI)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        final_loss = loss_fn(logits, Y_MULTI).item()
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return model, initial_loss, final_loss, probabilities, predictions


def print_three_class_extension():
    """Report probabilities, classes and a numerical softmax-sum check."""
    print("\n=== D. Three-class extension (2 -> 2 -> 3) ===")
    model, initial_loss, final_loss, probabilities, predictions = train_three_class()

    print(f"Final layer weight shape: {tuple(model.fc2.weight.shape)}")
    print("Number of logits per example: 3")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print("\nInput      Class probabilities [P(0), P(1), P(2)]                 Predicted   Target")
    for x, probs, pred, target in zip(X, probabilities, predictions, Y_MULTI):
        prob_list = [round(v, 6) for v in probs.tolist()]
        print(f"{tuple(int(v) for v in x.tolist())!s:<10} {str(prob_list):<52} {pred.item()}           {target.item()}")

    probability_sum = probabilities[1].sum().item()
    print(f"\nSoftmax sum for input (0, 1): {probability_sum:.8f}")

    assert torch.equal(predictions, Y_MULTI), "Three-class network did not learn all four examples."
    assert abs(probability_sum - 1.0) < 1e-6, "Softmax probabilities should sum to 1."
    print("Checks: PASS - all 4 classes are correct and the shown probability vector sums to 1.")


def main():
    print("AI Neural Models Hands-on")
    print("Dataset: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0 for binary XOR")
    print_binary_baseline()
    print_symmetry_experiment()
    print_activation_experiment()
    print_three_class_extension()
    print("\nALL REQUIRED EXECUTABLE CHECKS PASSED.")


if __name__ == "__main__":
    main()
