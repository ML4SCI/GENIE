# AE features + tree readout

Train an autoencoder (AE) on SB, freeze it, then train a 50-model CWoLa tree
ensemble on its features. The final event score is the mean tree probability.

Use `Nsig=1000`, 30 subjets and seed 42 by default. AE training and validation
use adjacent SB (3.1–3.3 and 3.7–3.9 TeV); trees use SR/SB labels; testing uses
SR (3.3–3.7 TeV).

[Run commands](../README.md#5-ae-features--tree-readout) ·
[Results](results.md#9-ae-features--tree-readout-sb)

The shell scripts prepare subjettiness tables and pack the selected signals'
graphs before training. Graphs come from `scripts/prepare_data.sh`.

## 1. NetGe AE: reconstruction + physics feature prediction

1. Use a NetGe max-message / bilinear AE to reconstruct
   relative-pT node features and physical-edge features.
2. Train it on LHCO SB for 50 epochs, jointly with heads predicting mass/τ,
   EFPs and log τ ratios. Each prediction MSE has weight `0.1`, added to the
   reconstruction loss. Inputs and targets use training-SB standardization.
3. Freeze the LHCO-trained AE. Train a separate SR/SB tree ensemble for each signal.

Mass and τ training targets come from the KS subjettiness tables. τ measures
alignment with N subjet axes; use N=1–9 and β=0.5, 1, 2 for both jets.
EFP targets are calculated from the input graph's momentum fractions and
angular distances: eight connected energy-flow polynomials per jet.
The ratio head predicts log(τ₂/τ₁), log(τ₃/τ₂) and log(τ₄/τ₃), with β=1.

The default `ratios` version supplies 80 tree inputs:

| Features | Count | Source |
| --- | ---: | --- |
| Lighter jet mass and mass difference | 2 | Sum massless-subjet four-vectors from the graph |
| τ predictions | 54 | AE mass/τ head |
| EFP predictions | 16 | AE EFP head; targets are sums and absolute differences of the two jets' log EFPs |
| Log reconstruction errors | 2 | `log(r0) + log(r1)` and `log(r0) - log(r1)`, with each `r` = node MSE + edge MSE |
| Log τ-ratio predictions | 6 | AE ratio head |

Graph masses replace the two mass predictions, retaining the absolute momentum
scale removed by relative-pT inputs. The `predicted`, `graph_mass` and `errors`
variants share one AE without the ratio head and provide 72, 72 and 74 inputs.

## 2. Dual AE: NetGe and pair-attention

Train a separate pair of AEs for each signal:

1. Both reconstruct graphs and predict mass/τ and EFPs. NetGe also predicts
   eight Soft Drop features: mass and pT retention, splitting fraction and opening
   angle, combined across the two jets as sums and absolute differences.
   These targets come from the graphs (β=0, zcut=0.1). The attention encoder
   uses all subjet pairs; reconstruction still uses Unique-6 edges.
2. Train each AE for 50 epochs, then jointly train for another 50. InfoNCE
   compares their 70 τ/EFP predictions: the same event is a positive pair;
   other events in the batch are negatives. Both AEs receive gradients.
   Loss = both AE objectives + `0.03 × InfoNCE` (temperature `0.2`).
   The control omits InfoNCE and starts from the same pretrained weights.
3. Freeze both AEs and train the tree ensemble on 154 inputs:
   140 τ/EFP predictions + 8 Soft Drop predictions + 4 log-error features
   + 2 graph masses.

The contrastive weight comes from gradient norms measured on 16 LHCO SR training
batches of 256 events after AE pretraining:

$$
\lambda = 0.5\,
\frac{\mathrm{median}_b \|\nabla_\theta L_{\mathrm{AE}}^{(b)}\|_2}
     {\mathrm{median}_b \|\nabla_\theta L_{\mathrm{InfoNCE}}^{(b)}\|_2}
\approx 0.03.
$$

Here, $L_{\mathrm{AE}} = L_{\mathrm{NetGe}} + L_{\mathrm{Attention}}$ and
$\theta$ contains both encoders' parameters. The factor `0.5` makes the weighted
contrastive gradient about half the AE gradient. We round to `0.03` and keep it
fixed across signals and seeds.

## 3. Tree readout

Each of the 50 gradient-boosted classifiers splits the SR/SB pool equally for
training and early-stopping validation. Some events used to validate the trees
were also used to train the AE. Signal/background truth is used for evaluation,
not the training objectives.

`--runs 4` repeats this NetGe AE on four LHCO KS splits (runs 0–3), retraining the AE
and 50-member readout each time. AE seed stays 42; scaling reuses the original
LHCO training-SB statistics. This differs from changing AE seeds on a fixed split.
