# Experiment C — which graph layer? (latent 128, graph decoder)

**Question.** With a stronger autoencoder, does the choice of graph layer
matter? ChebNet K=2, GraphSAGE and DeepSets (no graph at all) are trained
with everything else identical.

The code and run tags say `v19` (the project's internal version number); in
this repository the experiment is v18.

---

## What changed from v17

| part | v17 (Experiment B) | this experiment |
|---|---|---|
| latent | 64 | **128** |
| pooling | StatsPool (sum, mean, max, std) | **mean**, with **log N** appended before the latent |
| decoder | 150 independent slots + mask | **count head** p(N\|z) + first N slots → **kNN graph on the decoded particles → the same graph layer as the encoder** |
| N in the decoder | predicted through the mask | **true N** while training; sampled from the count head when generating |
| loss | Sinkhorn (pT×10, blur 0.05) + mask + count | **Sinkhorn (pT×3, blur 0.02) + energy-flow EMD + count-head cross-entropy** |
| model selection | last epoch | **best validation epoch** (JetNet's validation split) |

Loss: `L = L_sinkhorn + 1·L_emd + 0.1·L_count`.

- **Sinkhorn** matches every particle equally on (η, φ, 3·pT).
- **Energy-flow EMD** weights particles by their momentum share, so the hard
  particles that set the jet mass count most. It is also the suite's EMD
  metric, so that one metric is not independent of training.
- **Count** trains p(N|z), needed to choose N when generating. The decoder
  always makes exactly N particles.


## Setup

Undirected kNN(6) graph, hidden size 256, 2 graph layers in the encoder and
the decoder, latent 128, 200 epochs, batch 512, learning rate 1e-3 (cosine),
seed 0. Only the graph layer changes:

| model | graph layer | reach | params (total) |
|---|---|---|---|
| ChebNet K=2 | spectral, 1 hop per layer | 2 hops | 1,518,813 |
| GraphSAGE | averages direct neighbours | 2 hops | 1,518,813 |
| DeepSets | none: each particle updated alone | 0 hops | 1,256,669 |

All three use the same stable layer (Euler step, anti-symmetric weights).
DeepSets is exactly ChebNet with K=1 (no neighbour term), and its decoder
skips the kNN graph.

---

## Results

78,882 test jets, scored at each model's best validation epoch. Paired
relative errors (lower is better), as defined in the main README; the
group means use the main README's grouping. (`primary_mean` in these
`suite.json` files was computed by a cluster copy of `suite.py` that also
averaged in log D2, so it reads 0.109 / 0.110 / 0.109.)

| | ChebNet K=2 | GraphSAGE | DeepSets |
|---|---|---|---|
| ECF(2, β=1) | **0.091** | 0.095 | 0.095 |
| wide-angle EEC | 0.082 | 0.083 | 0.082 |
| **pairwise mean** | **0.087** | 0.089 | 0.088 |
| girth | 0.064 | 0.066 | 0.064 |
| jet mass | **0.099** | 0.104 | 0.103 |
| ECF(2, β=2) | **0.081** | 0.087 | 0.087 |
| **per-particle mean** | **0.082** | 0.085 | 0.085 |
| ptD (control) | **0.057** | 0.063 | 0.063 |
| log D2 (diagnostic) | 0.154 | 0.153 | **0.150** |
| energy-flow EMD | 0.0111 | 0.0113 | **0.0105** |
| count head: exact N | 96.4% | **99.9%** | **99.9%** |
| best epoch | 190 | 200 | 200 |
| training time (A100-40GB) | 7.6 h | 7.3 h | 6.7 h |
| decoder, generation mode | 29 µs/jet | 27 µs/jet | 16 µs/jet |

**Takeaway: at 2 layers the three models are tied on reconstruction.**
DeepSets, with no message passing at all, matches both graph models on the
pairwise quantities. ChebNet K=2 is slightly ahead on the pairwise *and* the
per-particle quantities and the control together.

## What's in here

```
code/
    ae_v19.py        train -> best validation epoch -> score the test split
                     (--conv cheb / sage / deepsets picks the graph layer)
    slurm_v18.sh     the three runs as a SLURM array, one GPU each
results/runs/<tag>/
    suite.json           all metrics, parameter counts, count-head accuracy, timing
    log.txt              full training log
    loss_terms.png       Sinkhorn, EMD, count and total loss, train and validation
    recon_pairs.png      six real test jets next to their reconstructions
                         (the same six jets for every model)
    dist_comparison.png  particle-level distributions, real vs reconstructed
    observables.png      jet-level observables, real vs reconstructed
```

## Running

`ae_v19.py` reuses the layers, data loading, graph caches and metric suite
from Experiment B (`../experiment_latent64_reach_v17/code/ae_core.py` and
`suite.py`); the SLURM script points `V17_DIR` there. Set the other paths at
the top of `slurm_v18.sh`, then from `code/`:

```bash
mkdir -p logs
sbatch --array=0 slurm_v18.sh      # ChebNet K=2; builds the graph caches
sbatch --array=1-2 slurm_v18.sh    # GraphSAGE and DeepSets, in parallel
```

A single run:

```bash
export V17_DIR=../../experiment_latent64_reach_v17/code
python ae_v19.py --tag demo --conv cheb --K 2 --layers 2 --latent-dim 128 --skip-ceiling
```

Requirements as in the main README.
