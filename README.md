# Variance-Aware Prior-Based Tree Policies for Monte Carlo Tree Search

This repository accompanies the preprint **Variance-Aware Prior-Based Tree Policies for Monte Carlo Tree Search**

[https://arxiv.org/abs/2512.21648](https://arxiv.org/abs/2512.21648)

It contains reference implementations and experimental code used in the paper, with a focus on variance-aware tree policies for Monte Carlo Tree Search (MCTS).


## Abstract

> Monte Carlo Tree Search (MCTS) has profoundly influenced Reinforcement Learning (RL) by integrating planning and learning in tasks requiring long-horizon reasoning, exemplified by the AlphaZero family of algorithms. Central to MCTS is the search strategy, governed by a tree policy based on an upper confidence bound (UCB) applied to trees (UCT). A key factor in the success of AlphaZero is the introduction of a prior term in the UCB1-based tree policy PUCT, which improves exploration efficiency and thus accelerates training. While many alternative UCBs with stronger theoretical guarantees than UCB1 exist (e.g., UCB-V), extending them to prior-based UCTs has been challenging, since PUCT was derived empirically rather than from first principles. Recent work retrospectively justified PUCT by framing MCTS as a regularized policy optimization (RPO) problem. Building on this perspective, we introduce Inverse-RPO, a general methodology that systematically derives prior-based UCTs from any prior-free UCB. Applying this method to the variance-aware UCB-V, we obtain two new prior-based tree policies that incorporate variance estimates into the search. Experiments show that these variance-aware prior-based UCTs outperform PUCT across multiple benchmarks, without incurring additional computational cost. We also release an extension of the mctx library supporting variance-aware UCTs, showing that the required code changes are minimal and intended to facilitate further research on principled prior-based UCTs.

## Repository Structure

This repository includes **copies** of two existing libraries, modified locally to support the methods proposed in the paper:

### pgx

Game environment library with AlphaZero-style training utilities: [https://github.com/sotetsuk/pgx](https://github.com/sotetsuk/pgx)

* `pgx/examples/alphazero/train_minatar.py`
  Script for training MinAtar games using the proposed tree policies.
* `pgx/examples/alphazero/Dockerfile`
  Dockerfile for reproducible MinAtar AlphaZero experiments.

### mctx

Monte Carlo Tree Search library: [https://github.com/google-deepmind/mctx](https://github.com/google-deepmind/mctx)

Key modified components:

* `mctx/_src/action_selection.py` – implementations of the proposed tree policies
* `mctx/_src/search.py` – core MCTS search logic
* `mctx/_src/policies.py` – RL policies using MCTS
* `mctx/_src/tree.py` – tree data structures
* `mctx/_src/base.py` – shared interfaces and types
* `mctx/_src/qtransforms.py` – value and variance normalization

All significant deviations from the upstream versions are explicitly marked with:

```python
# CHANGES FOR SUBMISSION START HERE
# CHANGES FOR SUBMISSION END HERE
```

## Provenance and Upstream Status

* The `pgx` and `mctx` directories are **copied and modified** from their respective upstream repositories for the purpose of the paper and experiments.
* This repository is **not** an official fork.
* In the future, selected components—particularly the variance-aware and Inverse-RPO-derived tree policies—may be proposed upstream as pull requests to `mctx`.

## Citation

If you use this code or build on the ideas, please cite the associated preprint.
