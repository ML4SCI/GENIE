# NetGe: variations and what we learned

We adapted NetGe from node/adjacency reconstruction to jet-feature reconstruction,
then tested input coordinates, edge decoders, message aggregation and event scoring.

[Reported SR settings](results.md#reported-sr-settings) ·
[Results overview](results.md#4-netge-jet-max-message--bilinear-decoder)

Tables show **AUC / maxSIC** at epoch 50, with SR training, `Nsig=1000`
and 30-subjet graphs.
The early experiments train one LHCO model and test it on all signals. The final
comparison trains a separate model for each signal, using seed 42.

## Retained variants

The four options in [netge.py](../src/models/netge.py):

| Variant | What changes |
| --- | --- |
| `original` | Node `ln(pT)` and binary adjacency; sum propagation, 128D node latent, node and adjacency reconstruction. Original cycle and contrastive training. |
| `jet` | Keep Unique-6 connections; jointly encode nodes and physical edges, using a biased embedding, a self-node path and mean messages. Reconstruct node and edge features from a 2D node latent. |
| `fraction` | Use `ln(pT_i / sum(pT))` within each jet. Standardize node and edge features; keep mean messages and the MLP edge decoder. |
| `max_bilinear` | Standardized `ln(pT)` and physical edges; max messages and a symmetric linear + bilinear edge decoder. Our retained main variant. |

`fraction` uses reconstruction-only training. `jet` and `max_bilinear` default
to reconstruction and support `--objective cycle`, `contrast`, or `full`.
`max_bilinear` defaults to geometric event scoring; `--score reconstruction`
selects the sum of both jet scores. This does not change training.

## 1. From adjacency to physical-edge reconstruction

With single-feature `ln(pT)` inputs, the original bias-free encoder produced
nearly collinear node vectors. Its inner-product adjacency decoder saturated
near one. Adding physical edges to that same architecture did not solve the problem.

The Jet adaptation instead keeps the graph connections and reconstructs their
physical features. Messages combine `h_i`, `h_j - h_i` and `e_ij`, with a separate
self-node path. This changes both the architecture and reconstruction task.

These early runs use **seed 123**, Adam `1e-4` and 150 events per batch. Original
and Edge use the cycle score; Jet uses node + edge MSE. Both jet scores are summed.

| Model / jet score | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original / cycle | 0.6819 / 1.055 | 0.6040 / 1.040 | 0.7152 / 1.153 | 0.9449 / 2.884 | 0.9825 / 6.688 |
| Edge / cycle | 0.6406 / 1.031 | 0.5419 / 1.001 | 0.6378 / 1.032 | 0.2109 / 1.010 | 0.2118 / 1.011 |
| Jet / reconstruction | 0.8999 / 2.149 | 0.8584 / 1.750 | 0.8535 / 1.731 | 0.8679 / 1.784 | 0.8553 / 1.719 |

Jet closes much of the gap on LHCO, WR and YY′. Original remains stronger on
Z′ and YHH; the adaptation does not improve every signal. The Edge version was archived.

## 2. Absolute versus relative momentum

Standardize features using training-set means and standard deviations. Then
replace node `ln(pT)` with `ln(pT_i / sum(pT))`, leaving physical edges unchanged.
The Fraction score is insensitive to a common momentum rescaling within a jet.

This matched pair uses **seed 123**, the default AdamW/OneCycleLR recipe, mean
messages, an MLP edge decoder and reconstruction-only training. Event scores sum
the two jets' node + edge MSE.

| Node coordinates | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standardized ln(pT) | 0.9015 / 2.292 | 0.8647 / 1.853 | 0.8579 / 1.843 | 0.8487 / 1.652 | 0.8545 / 1.680 |
| Fraction | 0.9019 / 2.291 | 0.8687 / 1.883 | 0.8585 / 1.835 | 0.8580 / 1.718 | 0.8627 / 1.741 |

Fraction gives small AUC gains on all five signals, but maxSIC does not improve
on LHCO or YY′. It remains a useful input control, not the strongest overall model.

## 3. A small decoder and stronger neighbor messages

Replace the edge MLP with a symmetric linear decoder, then add bilinear
interactions between the two endpoint vectors. The bilinear term adds just
**nine parameters**, allowing the decoder to use how the two latent coordinates
pair across an edge.

With mean messages and seed 123, linear → bilinear raises Z′ AUC
**0.8875 → 0.9030** and YHH **0.8486 → 0.8740**. Full-range BB1 falls slightly,
**0.8953 → 0.8943**.

Next, keep the bilinear decoder and change mean to channel-wise max aggregation
in both encoder layers. This changes neighbor messages, not Global Pooling.
The following matched **seed-42** runs use the default training recipe and sum scoring.

| Message aggregation | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean | 0.9028 / 2.169 | 0.8785 / 1.938 | 0.8604 / 1.801 | 0.9054 / 2.127 | 0.8761 / 1.845 |
| Max | 0.9129 / 2.480 | 0.8871 / 2.078 | 0.8700 / 1.920 | 0.9185 / 2.311 | 0.8971 / 2.010 |

Both metrics improve on all five signals. We retain this **26,174-parameter**
max-message / bilinear model with a two-dimensional node latent.

## 4. Other controls we did not retain

These seed-123 controls start from the standardized mean/MLP model above
(LHCO AUC **0.9015**), still training on LHCO only.

| Change | LHCO AUC | Outcome |
| --- | ---: | --- |
| Learned attention instead of mean messages | 0.8954 | AUC falls on all five signals; lower reconstruction loss does not translate into better anomaly ranking. |
| One 8D vector per jet instead of node latents | 0.8012 | AUC falls on all five signals, especially Z′ and YHH. |
| Add adjacency BCE to training and scoring | 0.7329 | The structure head learns, but the combined anomaly score gets worse on all five signals. |
| Gaussian input noise, reconstruct clean features | 0.9001 | Small gains on Z′ and YHH, losses on LHCO, WR and YY′. |

## 5. Final per-signal results and event scoring

Train the retained `max_bilinear` model separately for each signal under the
[reported SR settings](results.md#reported-sr-settings), seed **42**.
Use node + physical-edge MSE only.

For jet errors `s0` and `s1`, compare their sum with `2 * sqrt(s0 * s1)`.
The geometric score favors events where **both jets** reconstruct poorly.
The two NetGe rows use the same checkpoints.

| Model / event score | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| EdgeConv baseline / sum | 0.9024 / 2.148 | 0.8752 / 1.873 | 0.8578 / 1.728 | 0.8869 / 1.940 | 0.8836 / 1.903 |
| NetGe-Jet / sum | 0.9131 / 2.465 | 0.8863 / 2.065 | 0.8685 / 1.903 | 0.9166 / 2.274 | 0.8927 / 1.966 |
| NetGe-Jet / geometric | 0.9239 / 2.966 | 0.8908 / 2.193 | 0.8793 / 2.175 | 0.9347 / 2.720 | 0.9170 / 2.265 |

Geometric scoring improves both metrics on all five tests. This is a scoring
gain on top of the architecture change, not another trained model.

Apply the final LHCO checkpoint to BB1:

| Event score | BB1 full range | BB1 SR (3.3–3.7 TeV) |
| --- | ---: | ---: |
| Sum | 0.9001 / 2.756 | 0.8478 / 1.891 |
| Geometric | 0.9017 / 3.519 | 0.8369 / 1.812 |

The gain does not extend to BB1 SR: requiring both jets to have large errors
reduces AUC and maxSIC there.

## What we learned

- Matching the reconstruction task to jet features mattered more than scaling
  the original losses.
- Relative momentum gives modest gains; bilinear decoding and max messages
  gave the most useful structural improvements.
- More capacity, lower reconstruction loss and extra objectives do not
  automatically improve anomaly detection.
- Keep architecture changes and score changes separate when comparing results.

Adding cycle and contrastive tasks to the final backbone is covered separately
in [NetGe–GLADC](netge-gladc.md).
