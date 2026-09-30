# Lab Answers

## Q1
The chain-rule decomposition converts the probability of a complete sequence into a sequence of next-token conditional probabilities. Generation is therefore possible one token at a time: predict/sample the next token from the current context, append it, and repeat.

## Q2
The first-order Markov assumption is that the next token is conditionally independent of all earlier tokens once the immediately previous token is known:

`P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})`.

## Q3
The program prints the required conditional probability tables for `the`, `cat`, `dog`, `sat`, and `ran`. A transition not observed in the dataset has probability zero in this unsmoothed model.

## Q4
Transition counts are stored in `transition_counts`, implemented as `defaultdict(Counter)`.

## Q5
`P(X_t | X_{t-1})` is computed in `train()` by dividing each transition count by the total outgoing transition count for the current word.

## Q6
The code supports both methods. Greedy mode uses the token with maximum probability. Sampling mode draws according to the complete probability distribution. Greedy generation is deterministic for a fixed model, while sampling can produce different valid continuations.

## Q7
If no transition has been observed for a context, `distribution()` returns an empty dictionary and prediction/sampling returns `None`, so generation stops safely.

## Q8
A total of `0.87` means the conditional distribution has not been normalised correctly. Some probability mass is missing or the denominator/counting logic is wrong.

## Q9
No. The model only reflects patterns in the small training dataset, while human expectations use far more linguistic and world knowledge.

## Q10
Sampling produces more variation because it can choose any token with non-zero probability. Greedy generation always chooses an `argmax` token and therefore repeatedly follows the same highest-probability path.

## Q11
The second-order model:
1. conditions each token on two preceding tokens instead of one;
2. uses pair contexts `(X_{t-2}, X_{t-1})` in its conditional probability table;
3. has more context available for prediction;
4. needs more training data because many more contexts are possible and therefore sparsity increases.

## Q12
More context can improve prediction because it distinguishes situations that look identical to a shorter-context model. However, it increases the size of the conditional probability table, so each specific context is observed fewer times and many contexts may never appear in limited data.

## Q13
Approach B is preferable because it specifies the intended probabilistic behaviour before implementation. This makes the representation and assumptions explicit, lets us inspect whether the code matches the model, and allows testing of invariants such as each conditional distribution summing to 1.

## Q14
Thinking of the language model as a Bayesian network provides:
- an explicit representation of dependencies between tokens;
- a factorisation of the joint sequence probability;
- a clear interpretation of next-token conditional probabilities;
- a principled sampling procedure for generation;
- explicit independence assumptions;
- a way to reason about what changes when more context is used;
- testable conditions for checking whether code implements the intended probabilistic model.

## Short LLM Reflection
The LLM was used to translate a behavioural specification into ordinary Python code using transition counts and random sampling. The generated implementation was validated by inspecting where counts were stored, checking that probabilities were computed by normalising outgoing counts, and running a test that verifies every conditional distribution sums to approximately `1.0`.

One implementation detail I explicitly checked was handling unseen contexts. Instead of calling `random.choices()` on an empty distribution, the final code first checks whether a distribution exists and returns `None` when it does not. This prevents an exception and makes the behaviour for zero-probability contexts explicit.
