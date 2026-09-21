# ScaSAN and SybilScaRepAct

This repository contains the official source code, experiment configurations, tests, and supplementary materials for the manuscript:

> **Exploring Social-activity Networks for Sybil Detection: Models and Efficient Algorithms**

**Authors:** Zongyuan Chen, Tao Tan, Mingze Zhong, Hong Xie, and John C. S. Lui
**Corresponding author:** Tao Tan
**Venue:** IEEE ICASSP 2027 submission

## Paper and Resources

* [Full-version paper](./ScaSAN_Full_Version.pdf)
* [Latest release](https://github.com/CreeperKnight/ScaSAN-SybilScaRepAct/releases/latest)
* [Source code](./src/scasan)
* [Experiment configuration](./configs/paper.yaml)
* [Automated tests](./tests)

## Overview

Fake-account, or Sybil, detection is important for protecting online social networks from malicious activities. This project provides the implementation of ScaSAN and SybilScaRepAct presented in the paper.

**ScaSAN** is a scalable social-activity network that aggregates social activities according to their types. Unlike activity-per-node representations, ScaSAN contains at most three times as many nodes as users.

**SybilScaRepAct** is a random-walk-based Sybil-detection algorithm that evaluates user trust scores from a small set of labeled seed nodes. It mitigates the trapping-node problem by redistributing a small amount of trust to trusted seeds.

The implementation uses SciPy sparse matrices and matrix-free propagation, allowing the complete graph to be processed without constructing a dense 3N×3N3N \times 3N transition matrix.

## Main Features

The repository includes:

* the weighted Scalable Social-Activity Network (ScaSAN);
* the SybilScaRepAct trust-propagation algorithm;
* the SybilSocActNet baseline, called Sybil_SAN in its original paper;
* a friendship-only Sybil-detection baseline;
* downloading and deterministic preprocessing of the Facebook WOSN 2009 dataset;
* construction of synthetic Sybil regions;
* friendship, incoming-activity, and outgoing-activity attacks;
* parameter sweeps used in the paper and its full version;
* generation of the experimental figures;
* tests for probability-mass conservation, convergence, AUC, trapping nodes, attacks, and graph serialization.

## Repository Structure

```text
ScaSAN-SybilScaRepAct/
├── configs/
│   └── paper.yaml            # Default parameters and sweep settings
├── data/                     # Raw and processed dataset directories
├── scripts/
│   └── reproduce.sh          # End-to-end experiment script
├── src/scasan/
│   ├── algorithms.py         # Sybil-detection algorithms
│   ├── attacks.py            # Attack injection and seed selection
│   ├── experiments.py        # Experiment sweeps, timing, and plots
│   ├── graph.py              # Sparse graph representations
│   ├── metrics.py            # Tie-aware AUC computation
│   ├── preprocess.py         # Dataset downloading and preprocessing
│   └── synthetic.py          # Synthetic and toy graph construction
├── tests/                    # Automated tests
├── pyproject.toml            # Python project configuration
├── requirements.txt          # Python dependencies
├── ScaSAN_Full_Version.pdf   # Full-version manuscript
└── README.md
```

## Requirements

* Python 3.10 or newer
* NumPy
* SciPy
* Matplotlib
* PyYAML

No GPU is required.

## Installation

Clone the repository:

```bash
git clone https://github.com/CreeperKnight/ScaSAN-SybilScaRepAct.git
cd ScaSAN-SybilScaRepAct
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the project and its testing dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Run the automated tests:

```bash
pytest
```

## Quick Validation

The toy experiments do not require downloading the external Facebook dataset.

Run:

```bash
scasan toy \
  --config configs/paper.yaml \
  --output results/toy
```

This command generates the toy-graph experiments used to illustrate the trapping-node behavior:

* Figure 3(a): the AUC decreases from 1.01.0 at k=0k=0 to 0.50.5 when k≥3k \geq 3;
* Figure 3(b): the AUC remains 0.50.5 for n=1,…,10n=1,\ldots,10 when k=5k=5;
* Figure 11: directly applying SybilSocActNet to ScaSAN reaches an AUC of 0.50.5 for sufficiently large kk.

Unless otherwise stated, figure numbers in this README refer to the full-version manuscript.

## Complete Experiment Pipeline

Run the complete pipeline with:

```bash
./scripts/reproduce.sh
```

The script downloads and preprocesses the dataset, constructs the experimental graphs, runs the algorithms, and generates the metrics and figures.

The same pipeline can be executed using the following commands.

### 1. Download the Dataset

```bash
scasan download \
  --output data/raw
```

The downloader obtains the Facebook WOSN 2009 dataset from the dataset authors' official source. Downloads are performed atomically and validated as gzip files.

### 2. Preprocess the Dataset

```bash
scasan prepare \
  --links data/raw/facebook-links.txt.gz \
  --walls data/raw/facebook-wall.txt.gz \
  --output data/processed/facebook
```

### 3. Inspect the Processed Dataset

```bash
scasan inspect \
  --data data/processed/facebook
```

### 4. Run All Experiments

```bash
scasan run \
  --config configs/paper.yaml \
  --data data/processed/facebook \
  --output results/paper
```

## Running Selected Experiments

Individual parameter sweeps can be selected using the `--experiments` option.

For example:

```bash
scasan run \
  --config configs/paper.yaml \
  --data data/processed/facebook \
  --output results/selected \
  --experiments friendship_attack incoming_activity_attack runtime
```

The available experiment names are:

```text
friendship_attack
incoming_activity_attack
outgoing_activity_attack
alpha
beta
lambda1
lambda2
seed_count
runtime
```

Multiple experiment names can be supplied in a single command.

## Generated Outputs

The complete experiment pipeline produces the following files:

| Output file                    | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| `metrics.csv`                  | Experimental results in CSV format                             |
| `metrics.json`                 | Experimental results and configuration metadata in JSON format |
| `fig3_toy_sensitivity.pdf`     | Impact of kk and nn on the toy graph                         |
| `fig6_attack_strength.pdf`     | Friendship and incoming-activity attacks                       |
| `fig7_hyperparameters.pdf`     | Impact of α\alpha and β\beta                                 |
| `fig8_runtime.pdf`             | Runtime under two values of β\beta                            |
| `fig11_direct_application.pdf` | SybilSocActNet directly applied to ScaSAN                      |
| `fig12_outgoing_and_seeds.pdf` | Outgoing attacks and number of seed nodes                      |
| `fig13_activity_weights.pdf`   | Impact of λ1\lambda_1 and λ2\lambda_2                          |

The metrics files contain AUC values, parameter values, iteration counts, graph-construction time, and trust-propagation time.

Runtime plots report the trust-propagation time. Graph-construction time is retained separately in the metrics files.

## Data Preprocessing

Each wall-post record contains:

```text
wall_owner, poster, timestamp
```

For each wall, posts are sorted chronologically, and consecutive posts from the same user are merged into an activity block.

The activity blocks are processed as follows:

1. The first post of an odd block increments the user's created-content aggregate AiA_i.
2. The first post of an even block increments the user's forwarded-content aggregate aia_i and follows the activity in the preceding block.
3. The second post mentions the user associated with the preceding block.
4. The remaining posts like preceding activities in reverse chronological order.

The unaggregated graph retains one activity node per block for SybilSocActNet. ScaSAN stores at most one created-content aggregate and one forwarded-content aggregate for each user.

The preprocessing pipeline then:

* identifies the largest connected component of the friendship graph;
* connects nodes outside the largest component to selected nodes inside it;
* selects a consecutive 1,000-user ID window for constructing the Sybil region;
* duplicates the induced friendship and activity subgraphs;
* selects labeled honest seed users;
* injects friendship attacks;
* injects incoming-activity attacks;
* injects outgoing-activity attacks;
* saves the processed graph in a reusable sparse format.

Natural like edges are retained during preprocessing but excluded by the paper experiment runner for comparability with SybilSocActNet. Injected activity attacks are sampled from mention, forwarding, and like interactions.

## Algorithms

### SybilScaRepAct

SybilScaRepAct propagates trust over the ScaSAN graph from a small set of labeled seed nodes.

For a user node, the transition distributes:

* an α\alpha portion of its trust to honest seed users;
* a β\beta portion through weighted friendship edges;
* a 1−α−β1-\alpha-\beta portion through activity edges.

For a created-content activity node, the transition distributes:

* an α\alpha portion of its trust to honest seed activities;
* a 1−α1-\alpha portion to associated users.

For a forwarded-content activity node, the transition distributes:

* an α\alpha portion of its trust to honest seed activities;
* a γ\gamma portion through forwarding relationships;
* a 1−α−γ1-\alpha-\gamma portion to associated users.

The resulting user scores are normalized by weighted friendship degree. A larger normalized score indicates that the corresponding user is more likely to be honest.

The implementation supports:

* residual-based convergence;
* the theoretical iteration bound derived in the paper;
* sparse transition operations;
* trust-score normalization;
* configurable approximation accuracy.

Residual-based convergence is used by default. To use the theoretical bound, set:

```yaml
defaults:
  convergence: theoretical
```

in `configs/paper.yaml`.

### SybilSocActNet

The SybilSocActNet baseline implements the friendship, activity-following, and alternating user-activity random walks described in its original paper.

The implementation includes:

* source-dependent coupling;
* source-count normalization;
* friendship propagation;
* activity-following propagation;
* alternating user-activity propagation.

Its default parameters are:

```text
g = 0.15
k = 0
n = 1
```

### SybilFriendship

SybilFriendship is a friendship-only baseline based on a personalized random walk over the friendship graph.

It uses:

* a fixed restart probability of 0.10.1;
* trust propagation over friendship edges;
* friendship-degree normalization.

The restart probability remains fixed during the ScaSAN α\alpha and β\beta parameter sweeps.

## Default Configuration

All default values and parameter sweeps are defined in:

```text
configs/paper.yaml
```

The principal default parameters are:

| Parameter                       | Default value |
| ------------------------------- | ------------: |
| α\alpha                        |           0.1 |
| β\beta                         |          0.04 |
| γ\gamma                        |         0.425 |
| λ1\lambda_1                     |             2 |
| λ2\lambda_2                     |             3 |
| ϵ\epsilon                      |         0.001 |
| Number of honest seeds          |            10 |
| Number of friendship attacks    |         3,200 |
| Incoming-activity ratio ξ\xi   |       0.00002 |
| Outgoing-activity ratio ζ\zeta |          0.27 |

The number of independent experiment repetitions can also be configured in `configs/paper.yaml`. Results from each repetition are stored separately, and the generated plots report their mean values.

## Programmatic Usage

The algorithms can also be called directly from Python:

```python
import numpy as np

from scasan.algorithms import sybil_scarepact
from scasan.graph import ScaSANGraph

graph = ScaSANGraph.load("data/processed/facebook")
seeds = np.array([10, 42, 100])

result = sybil_scarepact(
    graph,
    seeds,
    epsilon=1e-3,
)

print(result.scores)
print(result.iterations)
print(result.propagation_seconds)
```

Each seed user must have at least one associated created-content or forwarded-content activity.

## Reproducibility

The repository centralizes all experiment parameters in `configs/paper.yaml`. Experiment outputs are stored in machine-readable CSV and JSON files together with their parameter values, iteration counts, and runtime measurements.

Random seeds are controlled by the configuration file, enabling deterministic preprocessing, attack construction, seed selection, and experiment execution.

To preserve an existing result directory, specify a new path with the `--output` argument.

## Data Availability

The experiments use the public Facebook WOSN 2009 dataset. The dataset can be downloaded through the provided command:

```bash
scasan download --output data/raw
```

The dataset is not redistributed in this repository and remains subject to the terms specified by its original authors.

## Data Citation

If you use the Facebook WOSN 2009 dataset, please cite:

```bibtex
@inproceedings{viswanath2009evolution,
  title     = {On the Evolution of User Interaction in Facebook},
  author    = {Viswanath, Bimal and Mislove, Alan and Cha, Meeyoung
               and Gummadi, Krishna P.},
  booktitle = {Proceedings of the 2nd ACM Workshop on Online Social Networks},
  year      = {2009}
}
```

## Citation

If you use this code or the ScaSAN model, please cite:

```bibtex
@misc{chen2026scasan,
  title  = {Exploring Social-activity Networks for Sybil Detection:
            Models and Efficient Algorithms},
  author = {Chen, Zongyuan and Tan, Tao and Zhong, Mingze and
            Xie, Hong and Lui, John C. S.},
  year   = {2026},
  note   = {Manuscript submitted to IEEE ICASSP 2027},
  url    = {https://github.com/CreeperKnight/ScaSAN-SybilScaRepAct}
}
```

The citation information will be updated after publication.

## License

The source code is released under the MIT License. See the `LICENSE` file for details.

The Facebook WOSN 2009 dataset remains subject to its original terms and is not covered by the repository's software license.
