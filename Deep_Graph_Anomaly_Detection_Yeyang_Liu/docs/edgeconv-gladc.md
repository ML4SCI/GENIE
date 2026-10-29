# EdgeConv–GLADC: what we tried and learned

We added GLADC's cycle and contrastive tasks to our EdgeConv baseline, then
tested neighbor messages, feature standardization and relative momentum inputs.

[Reported SR settings](results.md#reported-sr-settings) ·
[Results overview](results.md#3-edgeconvgladc-and-variations)

Tables show **AUC / maxSIC** at epoch 50 for SR-trained models.

Run one variant on SB (`original`, `neighbor`, `standardized`, `fraction`):

```bash
MODEL=gladc RUN_DIR=runs/gladc_fraction_sb bash scripts/train.sh --gladc-variant fraction
```

## 1. Add cycle and contrastive training

The question was whether extra training tasks would make reconstruction more
useful for anomaly detection. We kept the EdgeConv autoencoder and Unique-6
connections, and added:

- **Cycle:** reconstruct node and physical-edge features, then re-encode them.
  Compare node vectors and the jet vector from Global Pooling + projection
  before and after reconstruction.
- **Contrastive:** encode the same jet with the original encoder and a copy
  with Gaussian noise added to its weights. Train their vectors to be closer
  to each other than to those of other jets in the batch.

All added losses have weight `1`; anomaly scoring still uses reconstruction
error. We call this version **EdgeConv–GLADC**.

LHCO AUC improves slightly, **0.9024 → 0.9038**, but YHH drops from
**0.8836 → 0.8692**. Adding the tasks alone gives mixed results.

## 2. Give messages the neighboring node's features

Our baseline messages use the central node and physical-edge features. The
**Neighbor** control adds `h_j - h_i`, so a message can also use the difference
between the two nodes. Mean aggregation and the decoder stay unchanged.

Compared with EdgeConv–GLADC, AUC improves on LHCO (**0.9038 → 0.9087**) and YY′
(**0.8611 → 0.8644**), but falls on Z′ (**0.8855 → 0.8221**) and YHH
(**0.8692 → 0.8346**). More node information helps some signals but produces
a large trade-off on others.

## 3. Put node and edge features on comparable scales

The **Standardized** control starts from EdgeConv–GLADC and standardizes each feature
using its training-set mean and standard deviation. Both reconstruction and
cycle re-encoding use these coordinates. This tests feature scaling separately
from changing the messages.

YHH AUC rises from **0.8692 to 0.8748**; changes on the other signals are small.
Feature scaling alone gives only a modest improvement.

## 4. Describe momentum sharing within the jet

The **Fraction** control builds on Standardized. Replace node `ln(pT)` with
`ln(pT_i / sum(pT))`, using the sum within each jet. This tests whether the
distribution of momentum among subjets is more useful than its absolute scale.
Physical-edge features stay unchanged.

Compared with Standardized, AUC improves on WR (**0.8755 → 0.8795**), Z′
(**0.8851 → 0.8939**) and YHH (**0.8748 → 0.8816**). LHCO and YY′ change little.
Compared with EdgeConv–GLADC, Fraction improves both AUC and maxSIC on four of five signals.

## 5. Full comparison

Each model trains on its own signal's SR mixture, with `Nsig=1000` and seed 42.

| Model | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.9024 / 2.148 | 0.8752 / 1.873 | 0.8578 / 1.728 | 0.8869 / 1.940 | 0.8836 / 1.903 |
| EdgeConv–GLADC | 0.9038 / 2.178 | 0.8767 / 1.875 | 0.8611 / 1.763 | 0.8855 / 1.918 | 0.8692 / 1.793 |
| Neighbor | 0.9087 / 2.554 | 0.8670 / 1.910 | 0.8644 / 2.011 | 0.8221 / 1.499 | 0.8346 / 1.555 |
| Standardized | 0.9035 / 2.181 | 0.8755 / 1.872 | 0.8623 / 1.767 | 0.8851 / 1.916 | 0.8748 / 1.833 |
| Fraction | 0.9035 / 2.166 | 0.8795 / 1.909 | 0.8630 / 1.767 | 0.8939 / 1.990 | 0.8816 / 1.885 |

![EdgeConv–GLADC variations, baseline and KS ensemble: AUC and maxSIC](../assets/gladc.png)

## 6. Does changing the anomaly score help?

Using the same EdgeConv–GLADC checkpoints, compare four jet scores, then sum
the two jets:

- **Reconstruction:** node MSE + edge MSE.
- **Edge only:** edge MSE.
- **Cycle:** node-cycle MSE + graph-cycle MSE.
- **Reconstruction + cycle:** add the reconstruction and cycle errors.

| Score | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reconstruction | 0.9038 / 2.178 | 0.8767 / 1.875 | 0.8611 / 1.763 | 0.8855 / 1.918 | 0.8692 / 1.793 |
| Edge only | 0.9042 / 2.180 | 0.8771 / 1.879 | 0.8618 / 1.769 | 0.8862 / 1.926 | 0.8699 / 1.798 |
| Cycle | 0.7423 / 1.206 | 0.7239 / 1.158 | 0.7095 / 1.113 | 0.8434 / 1.593 | 0.8613 / 1.708 |
| Reconstruction + cycle | 0.9036 / 2.173 | 0.8764 / 1.872 | 0.8609 / 1.762 | 0.8858 / 1.919 | 0.8700 / 1.797 |

Edge-only scoring closely matches full reconstruction. Cycle alone is weaker;
adding it to reconstruction changes little.

## 7. Transfer to BB1

Apply each variant's LHCO-trained model to BB1. Full range contains 999,166
background and 834 signal events; SR (`3.3 < mJJ < 3.7 TeV`) contains 120,279
background and 223 signal events. Both use the reconstruction score and the
same maxSIC efficiency cut as the main comparison.

| Model | BB1 full range | BB1 SR |
| --- | ---: | ---: |
| EdgeConv–GLADC | 0.8854 / 2.461 | 0.8412 / 1.924 |
| Neighbor | 0.8707 / 2.437 | 0.8266 / 1.902 |
| Standardized | 0.8795 / 2.352 | 0.8351 / 1.818 |
| Fraction | 0.8859 / 2.358 | 0.8371 / 1.770 |

These variants do not improve BB1 SR. Fraction slightly raises full-range AUC,
but lowers maxSIC. SR results are calculated from the same saved event scores.

## What we learned

- Cycle and contrastive tasks do not consistently improve the baseline.
- Neighbor differences help LHCO and YY′, but hurt Z′ and YHH substantially.
- Fraction gives the broadest gains among these EdgeConv–GLADC variations on the five
  main tests, but does not improve BB1 SR.
- Cycle error works poorly as a standalone anomaly score.
