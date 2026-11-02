# NetGe–GLADC: cycle and contrastive training

Does adding GLADC's training tasks improve our strongest NetGe-Jet backbone?
We compare reconstruction-only training with reconstruction + cycle + contrastive.

Run on SB:

```bash
MODEL=netge RUN_DIR=runs/netge_gladc_sb bash scripts/train.sh --netge-variant max_bilinear --objective full
```

[Reported SR settings](results.md#reported-sr-settings) ·
[Results overview](results.md#4-netge-jet-max-message--bilinear-decoder) ·
[NetGe architecture variations](netge.md)

## 1. Keep the backbone and score fixed

- **Backbone:** NetGe-Jet with max-message aggregation, a two-dimensional node
  latent and a symmetric linear + bilinear edge decoder.
- **Inputs:** standardized node `ln(pT)` and physical-edge features on Unique-6
  graphs. Fit means and standard deviations on training data.
- **Jet score:** node MSE + physical-edge MSE.
- **Event score:** `2 * sqrt(s0 * s1)` for the two jets.

## 2. Add the GLADC tasks

- **Cycle:** reconstruct node and physical-edge features on the same connections,
  then re-encode. Compare node vectors and Global Pooling + projection vectors
  before and after reconstruction.
- **Contrastive:** encode the same jet with the original encoder and a copy with
  Gaussian noise added to its weights. Bring matching jet vectors closer than
  those of other jets in the batch.

Node cycle, graph cycle and contrastive losses each have weight `1`. We call
this version **NetGe–GLADC**; inference still uses the reconstruction score above.

## 3. Results

Separate SR training for each signal, `Nsig=1000`, seed 42. Entries are
**AUC / maxSIC** at epoch 50.

| Model | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| NetGe-Jet | 0.9239 / 2.966 | 0.8908 / 2.193 | 0.8793 / 2.175 | 0.9347 / 2.720 | 0.9170 / 2.265 |
| NetGe–GLADC | 0.9263 / 3.138 | 0.8867 / 2.109 | 0.8876 / 2.323 | 0.9151 / 2.259 | 0.8961 / 1.998 |

![NetGe scoring and auxiliary-loss comparisons, EdgeConv baseline and KS ensemble](../assets/netge.png)

## What we learned

- Both metrics improve on LHCO and YY′, but fall on WR, Z′ and YHH.
- The largest drops are Z′ AUC **0.9347 → 0.9151** and YHH **0.9170 → 0.8961**.
- We retain reconstruction-only NetGe-Jet with geometric scoring as the main
  variant; adding both tasks does not give a consistent improvement.
