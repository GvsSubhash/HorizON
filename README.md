# Multi-Horizon Deep Operator Network (HorizON)

HorizON is a framework for training deep operator network (DeepONet)-style models on **multi-fidelity**, **multi-property**, and **multi-horizon** data. A single model is trained jointly across heterogeneous data sources that differ in fidelity (e.g. experimental vs. theoretical), in the property being predicted (e.g. capacity vs. voltage), and in the operating horizon (e.g. cycle number) at which each data point was measured — rather than training a separate model per fidelity level, per property, or per horizon.

The architecture follows a branch–trunk DeepONet formulation: a **branch network** encodes the operating conditions (material representation, fidelity flags, processing conditions) into a latent embedding, a **trunk network** encodes the query horizon (e.g. cycle number) into the same latent space, and a dot product between the two — plus a small bias correction network conditioned on the processing conditions — produces the final prediction. This lets the model interpolate/extrapolate a predicted property continuously across the horizon axis for any given set of operating conditions, while sharing representational capacity across every fidelity level and property jointly.

## Contents

- **`HorizON.ipynb`** — trains the base HorizON model. Loads the multi-task dataset, builds a stratified 5-fold split, and runs an Optuna hyperparameter search over the branch/trunk/bias network architecture for each fold, saving one trained model per fold.
- **`Meta-learner.ipynb`** — builds an ensemble meta-learner on top of the 5 base HorizON models (one per fold) trained in `HorizON.ipynb`. Each base model produces stochastic (MC-dropout) predictions on the training and validation data; these 5 base-model prediction sets are stacked into a new feature representation and used to train a second-stage DeepONet-style meta-learner, again via an Optuna search, to combine the 5 folds' predictions into a single refined prediction.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

**Note:** this repository contains code only. No training data or trained model weights are included or required to read the pipeline; both are specific to our internal dataset and are not publicly distributed.
