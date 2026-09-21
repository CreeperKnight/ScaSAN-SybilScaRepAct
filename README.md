# ScaSAN: full experiment reproduction

This repository is a sparse reference implementation of the experiments in
**“Exploring Social-Activity Networks for Sybil Detection: Models and Efficient
Algorithms.”** The supplied paper is available at
[ScaSAN_Full_Version.pdf](ScaSAN_Full_Version.pdf).

The project includes:

- the weighted Scalable Social-Activity Network (ScaSAN);
- the paper's SybilScaRepAct/SybiSRA trust-propagation algorithm;
- the SybilSocActNet (called Sybil_SAN in its original paper) baseline;
- a friendship-only baseline;
- Facebook WOSN 2009 download and deterministic preprocessing;
- synthetic Sybil-region construction and all three attack families;
- every main-paper and appendix parameter sweep;
- generation of Figures 3, 6–8, and 11–13;
- tests for mass conservation, AUC, trapping nodes, attacks, and serialization.

The implementation uses SciPy sparse matrices and matrix-free propagation. It
runs the full graph without constructing a dense \(3N \times 3N\) matrix.

## Repository layout

~~~text
configs/paper.yaml          Paper defaults and all sweep values
scripts/reproduce.sh        End-to-end reproduction script
src/scasan/algorithms.py    All three detection algorithms
src/scasan/attacks.py       Attack injection and seed selection
src/scasan/experiments.py   Sweeps, timing, output, and plots
src/scasan/graph.py         Sparse graph containers
src/scasan/metrics.py       Tie-aware AUC
src/scasan/preprocess.py    Download, wall-post mapping, duplication
src/scasan/synthetic.py     Figures 1–4 and 10 toy graphs
tests/                      Automated tests
~~~

## Installation

Python 3.10 or newer is required. No GPU is needed.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[test]
pytest
~~~

The core dependencies are NumPy, SciPy, Matplotlib, and PyYAML.

## Quick validation without external data

Figures 3 and 11 use only the five-user graph from the paper:

~~~bash
scasan toy --config configs/paper.yaml --output results/toy
~~~

This reproduces the reported trapping-node behavior:

- Figure 3(a): AUC falls from 1.0 at \(k=0\) to 0.5 at \(k \ge 3\).
- Figure 3(b): AUC remains 0.5 for \(n=1,\ldots,10\) when \(k=5\).
- Figure 11: applying SybilSocActNet to aggregated ScaSAN also reaches
  AUC 0.5 for sufficiently large \(k\).

## Full reproduction

Run all stages with:

~~~bash
./scripts/reproduce.sh
~~~

Equivalent explicit commands are:

~~~bash
scasan download --output data/raw

scasan prepare \
  --links data/raw/facebook-links.txt.gz \
  --walls data/raw/facebook-wall.txt.gz \
  --output data/processed/facebook

scasan inspect --data data/processed/facebook

scasan run \
  --config configs/paper.yaml \
  --data data/processed/facebook \
  --output results/paper
~~~

The downloader uses the dataset authors' official
[WOSN 2009 data page](https://socialnetworks.mpi-sws.org/data-wosn2009.html).
Downloads are atomic and gzip-validated.

To run selected sweeps:

~~~bash
scasan run \
  --config configs/paper.yaml \
  --data data/processed/facebook \
  --output results/selected \
  --experiments friendship_attack incoming_activity_attack runtime
~~~

Valid names are friendship_attack, incoming_activity_attack, alpha, beta,
runtime, outgoing_activity_attack, seed_count, lambda1, and lambda2.

## Generated outputs

| File | Paper result |
|---|---|
| metrics.csv and metrics.json | Raw AUC, timing, iteration, and parameter values |
| fig3_toy_sensitivity.pdf | Impact of \(k\) and \(n\) |
| fig6_attack_strength.pdf | Friendship and incoming-activity attacks |
| fig7_hyperparameters.pdf | \(\alpha\) and \(\beta\) |
| fig8_runtime.pdf | Runtime at two \(\beta\) values |
| fig11_direct_application.pdf | Old algorithm applied to ScaSAN |
| fig12_outgoing_and_seeds.pdf | Outgoing attacks and seed count |
| fig13_activity_weights.pdf | \(\lambda_1\) and \(\lambda_2\) |

Runtime plots report propagation time; construction time is also retained in
the metrics files.

## Data transformation

Each raw wall-post row contains wall_owner, poster, and timestamp. For every
wall, posts are sorted chronologically and consecutive posts from the same
poster are merged into a block. Section 5.1 is then applied literally:

1. The first post of an odd block increments that user's created-content
   aggregate \(A_i\).
2. The first post of an even block increments forwarded-content aggregate
   \(a_i\) and follows the preceding block's activity.
3. The second post mentions the preceding block's poster.
4. Remaining posts like preceding activities in reverse chronological order.

The unaggregated graph keeps one activity per block for SybilSocActNet. ScaSAN
stores at most one created and one forwarded aggregate per user.

The code then:

1. connects each node outside the largest friendship component to a uniformly
   selected node inside it;
2. selects the consecutive 1,000-user ID window whose average activity count
   is closest to the global mean;
3. duplicates its induced friendship and activity subgraphs as the Sybil
   region;
4. injects friendship, incoming-activity, and outgoing-activity attacks using
   deterministic seeds.

Natural like edges are saved but removed by the paper experiment runner for
comparability with SybilSocActNet. Injected attack types are sampled uniformly
from mention, forwarding, and like as stated in the paper.

## Algorithms

### SybilScaRepAct

The implementation follows Algorithms 1–2. A transition distributes:

- \(\alpha\) user mass to honest seed users;
- \(\beta\) user mass through weighted friendships;
- \(1-\alpha-\beta\) user mass through activity edges;
- created-activity mass \(1-\alpha\) to associated users;
- forwarded-activity mass \(1-\alpha-\gamma\) to associated users;
- forwarded-activity mass \(\gamma\) through forwarding edges;
- activity mass \(\alpha\) to honest seed activities.

User scores are divided by weighted friendship degree. Residual convergence is
the default because the experiment section says it uses the same convergence
test as SybilSocActNet. The Corollary 1 bound is also implemented; set
defaults.convergence to theoretical to use it.

### SybilSocActNet

The baseline implements Algorithms 1–3 from
[Zhang et al., TDSC 2023](https://www.cse.cuhk.edu.hk/~cslui/PUBLICATION/TDSC-23a.pdf):
the friendship, activity-following, and alternating user–activity walks,
source-dependent coupling, and source-count normalization. Defaults are
\(g=0.15\), \(k=0\), and \(n=1\).

### SybilFriendship

The ScaSAN paper describes this comparison only as a friendship-only variant
and gives no transition equation. This repository uses a personalized
friendship random walk with fixed restart 0.1 and degree normalization. Its
restart remains fixed in ScaSAN \(\alpha\) and \(\beta\) sweeps.

## Configuration

All values live in [configs/paper.yaml](configs/paper.yaml).

| Parameter | Default |
|---|---:|
| \(\alpha\) | 0.1 |
| \(\beta\) | 0.04 |
| \(\gamma\) | 0.425 |
| \(\lambda_1\) | 2 |
| \(\lambda_2\) | 3 |
| \(\epsilon\) | 0.001 |
| honest seeds | 10 |
| friendship attacks | 3,200 |
| \(\xi\) | 0.00002 |
| \(\zeta\) | 0.27 |

Increase repeats to average independent attack and seed draws. Each repeat is
stored separately and plots show the mean.

## Reproducibility notes and paper ambiguities

The paper provides no source code, random seed, every interior sweep point, or
all boundary-case transition rules. This repository makes these decisions
explicit:

- Unprinted sweep points are inferred from figure axes and geometric patterns.
- Figure 13 includes \(\lambda_1=1\), while the method states
  \(\lambda_2>\lambda_1>1\). The code permits
  \(\lambda_2>\lambda_1\ge1\).
- Dummy aggregate activity nodes are omitted, matching the paper's “at most
  \(3N\)” statement and avoiding zero transition rows.
- If a forwarded aggregate has no outgoing forwarding edge, its otherwise
  undefined \(\gamma\) mass is sent to seed activities.
- Repeated activity events add weight; friendship edges stay binary.
- The friendship-only baseline uses the fixed-restart definition above.

Using the official files, raw counts are 1,545,686 friendship records and
876,993 wall posts with 63,891 users. This deterministic interpretation
produces 557,627 honest activity blocks and 8,732 copied Sybil blocks. With
natural likes removed, \(V_{HH}=400{,}963\) and \(V_{SS}=350\). The paper
reports 590,262 honest activities, 6,187 Sybil activities,
\(V_{HH}=429{,}547\), and \(V_{SS}=318\). Exact equality cannot be guaranteed
without the unavailable preprocessing code and tie/boundary rules. Audit local
counts with scasan inspect.

A validated default run on the full processed graph produced AUCs of about
0.887 (SybilScaRepAct), 0.701 (SybilSocActNet), and 0.625
(SybilFriendship), matching the ordering and scale in the paper. Runtime
depends on hardware.

## Programmatic use

~~~python
import numpy as np
from scasan.algorithms import sybil_scarepact
from scasan.graph import ScaSANGraph

graph = ScaSANGraph.load("data/processed/facebook")
seeds = np.array([10, 42, 100])
result = sybil_scarepact(graph, seeds, epsilon=1e-3)

print(result.scores)
print(result.iterations, result.propagation_seconds)
~~~

Seeds must have at least one created or forwarded activity.

## Tests

~~~bash
pytest
~~~

Tests cover probability-mass conservation, convergence, Figures 3 and 11,
tie-aware AUC, deterministic non-mutating attacks, and graph round trips.

## Data citation

Bimal Viswanath, Alan Mislove, Meeyoung Cha, and Krishna P. Gummadi.
“On the Evolution of User Interaction in Facebook.” WOSN, 2009.
