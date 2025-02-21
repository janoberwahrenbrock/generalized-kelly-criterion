# Generalized Kelly Criterion

## Introduction

This project implements the **Generalized Kelly Criterion**, extending the classical Kelly approach to handle not only binary outcome games but also games where outcomes lie on a continuous interval or belong to a discrete set.

## What is the Kelly Criterion?

The **Kelly Criterion** determines the optimal position size per round to maximize the long-term growth rate of capital. Instead of focusing on expected value, it considers logarithmic utility, ensuring that the capital grows at the highest possible compounded rate over multiple rounds.

For **binary outcome games** (e.g., betting on a biased coin flip), the Kelly fraction \( f^* \) is given by:

\[
f^* = \frac{P}{A} - \frac{Q}{B}
\]

where:

- \( P \) is the probability of winning,
- \( Q = 1 - P \) is the probability of losing,
- \( A \) is the return multiplier for a win (i.e., for every unit bet, you receive \( A \) additional units if you win),
- \( B \) is the loss multiplier (i.e., for every unit bet, you lose \( B \) units if you lose).

However, real-world problems are often more complex than simple binary bets. The **Generalized Kelly Criterion** extends this formula to handle cases where outcomes belong to a set of discrete possibilities or a continuous range.

To understand the mathematical derivation and the notations used in this project, it is strongly recommended to read **"Derivation and Application of the Generalized Kelly Criterion"** (provided in this repository).

---

## File Overview

- **`generalizedKellyCriterion.py`**  
  Implements the **Generalized Kelly Criterion**, allowing calculations for both continuous and discrete outcome spaces.

- **`examples.py`**  
  Provides example applications of the Generalized Kelly Criterion, including:  
  - A manipulated coin toss game (discrete outcomes).  
  - A Gaussian-distributed return model (continuous outcomes).

- **`requirements.txt`**  
  Contains all necessary dependencies for running the implementation.

- **`derivation_and_application_of_generalized_kelly_criterion.pdf`**  
  A detailed mathematical explanation of the **Generalized Kelly Criterion** and its applications.

---

## Installation

To use this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

After installation, you can look into `examples.py` to see how to use this implementation of the **Generalized Kelly Criterion**:

```bash
python examples.py
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For any questions or contributions, feel free to open an issue or submit a pull request.
