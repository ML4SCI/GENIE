# Flow-matching track: report at the technical end date

GSoC 2026, ML4SCI / GENIE, PINNDE project. 
15 June to 22 September 2026.

---

## 1. What the project is for

Simulating calorimeter showers with Geant4 is accurate and slow. The goal is a
generative model that produces showers of the same quality much faster.

This track uses **conditional flow matching**. The model learns a velocity
field: given a point and a time, it predicts which way to move. Training draws a
real shower and a noise vector, picks a random point on the straight line
between them, and asks the network for the direction along that line. Generating
starts from noise and follows the field. The target is exact and needs no
simulation during training, and the straight paths mean few steps are needed to
generate.

The evaluation module below was written to be usable by any generative model
on this data, not only this one.

Data is CaloChallenge dataset 2: two files of 100,000 showers, 6,480 voxels each
(45 layers × 16 angular × 9 radial), incident energies log-uniform from 1 GeV to
1 TeV. We train on one file and score against the other.

## 2. What was built

Two packages: 31 modules, 5,178 lines of code, 2,238 lines of tests, 171 tests
passing.

**`pinnde_eval/`** is the evaluation module. One call compares any two sets of
showers with three tiers of metrics: MMD and sliced Wasserstein for monitoring
during training; the CaloChallenge's own classifier AUC, chi2 and separation
power; FPD, KPD, Sinkhorn, and the classical two-sample tests (KS,
Cramér-von Mises, Anderson-Darling) with a Bonferroni correction. Also:

- `calochallenge.py`: the challenge's own 362 high-level features, computed
  with their code, pinned by commit and checksum, checked against their own
  classifier input to a relative tolerance of 1e-12.
- `floors.py` and the `validate_*` scripts: what a **perfect** generator
  scores, which is not zero. Every number in this project is quoted against a
  measured Geant4-against-Geant4 floor.
- `validate_physical.py`: nine rules asking whether a single generated shower
  could exist, with real showers run through the same code as a control.
- `diagnose_samples.py`: the same scores per observable family, which is how we
  find out *what* is wrong rather than only that something is.

**`flow_matching/`** is the generator: velocity field, training loop, Euler and
Heun samplers, and two entry points. `demo_calo.py` generates the 362 features
directly. `demo_voxels.py` generates the 6,480 voxels and computes the features
from them with the challenge's code.

## 3. What was found

**The floor has to be energy-matched.** Our model generates at the evaluation
set's own energies, so it never pays the energy-draw variance two independent
Geant4 samples pay. Matched and unmatched floors differ by 3 to 5 standard
errors on chi2, swd, w1 and separation power. Scored against the wrong one, a
model once looked better than Geant4.

**Separation power is more sensitive than the classifier.** On known energy
shifts, separation power clears its floor at a shift of about 0.09 in mean
log E_inc; the classifier AUC needs about 0.5. An AUC near 0.5 on its own is
weak evidence.

**The main failure was a point mass, not model capacity.** An empty layer is an
exact value in the official features: log10 E = −8, centres and widths 0,
sparsity 1. That is 17.8% of all layer-shower pairs and 51% of the last layer. A
continuous density puts zero probability on any exact value, so the model
produced an exactly-empty layer **0.0% of the time** at any capacity. It was not
wrong about *how often* layers are empty, since 18.5% of its mass sat in the
no-energy region. It had no way to say so, and spilled a third of that below
−8, which means less energy than nothing.

Giving the point mass a band to land in during training and snapping it back
when sampling, over five seeds each:

```
                        chi2 vs floor     separation vs floor        AUC
  baseline              50.7 +- 3.4          52.8 +- 3.5      0.909 +- 0.017
  with the fix          11.7 +- 2.3          11.1 +- 2.3      0.894 +- 0.020
```

For comparison, going from a 384×5 network to 1024×8 cost 30 times the compute
and gained 0.09 in AUC. The representation change cost nothing.

**Constraints belong to the physical object, not the number.** Snapping each of
the 362 features independently made the model *worse* (AUC 0.892 → 0.968): it
produced layers with no energy but a non-zero centre of energy. The rule has to
run per layer, and in both directions: a layer with energy must have at least
one lit voxel. That second direction took energy-in-zero-voxels from 58.6% of
showers to 5.9%, and sparsity-out-of-range from 44.4% to 7.0%, which had been
listed as a separate problem.

**Match the scale the evaluation uses.** Modelling layer energies as
√(E/E_inc) removed every impossible negative energy and scored far worse
(AUC 0.992). A miss of 0.015 in √-space, which is negligible there, is seven log
units away in the space the metrics read.

**Physical validity is a separate axis from distributional agreement.** The
checks found defects no metric had seen: every generated shower had a total
energy that was not the sum of its layers, and 63% had a layer holding energy in
zero lit voxels. The first is fixed exactly by computing the total from the
layers instead of modelling it.

**Measure the noise before believing a comparison.** Five seeds of one
configuration give AUC ±0.020 and chi2 ±20%. Estimated from two runs earlier in
the project, the spread looked much smaller, and several single-run numbers
quoted to three figures were too precise. One conclusion did not survive the
correction.

**The marginals, not the correlations, are the binding constraint.** The natural
reading of "chi2 improved fourfold, the classifier did not move" is that what
remains is cross-layer correlation. Install this model's marginals into Geant4's
exact correlation structure and the result is still detectable at about 0.92,
against about 0.93 for the model itself. Correlations add roughly 0.01 on top of
the marginal error, so per-feature accuracy is what the classifier reads.
`pinnde_eval/validate_claims.py` reproduces this from a saved sample file, and
the exact digits move by a few thousandths depending on which run it is given.

That test is only well posed where no point masses couple the columns together,
so it was run on the front ten layers, which are almost never empty. Those are
also the layers where the empty-layer fix does the least. Whether correlations
dominate in the deep layers is untested, and the same method cannot answer it
there.

## 4. What does not work yet

**The classifier AUC has never moved.** 0.909 ± 0.017 before, 0.894 ± 0.020
after. Every gain is in the marginal metrics and in physical validity.

**Voxel generation.** `demo_voxels.py` makes every consistency rule hold
automatically, because the features are computed from a voxel grid rather than
predicted as separate numbers. It does not work: in every run tried, each generated shower deposits
more than ten times the energy it received. The log representation has no
ceiling, 22.4% of generated layers hold more energy than any layer in the real
data, and those layers carry essentially all of the model's energy. More data
and more training changed nothing, so this is not undertraining.

**One result is unexplained.** A rank-Gaussian feature transform makes every
marginal exactly right by construction. Over three seeds it halves chi2 (11.7 to
5.9 times the floor), collapses the seed spread from 2.3 to 0.1, and removes
every impossible value, the best physical validity measured. It also makes the
classifier worse, 0.894 to 0.956. The obvious explanation, noise from dithering
tied values, was tested and ruled out at 1 to 7 percent of variance. The cause
is not identified. This is three seeds against five for the numbers above, and
it was measured in the last days of the program, so it is the least settled
result here.

## 5. Reproducing it

The data is not in the repository (1.36 GB per file). `cluster/get_data.sh`
downloads both files from Zenodo and checks their MD5s. Tests that need the data
skip themselves when it is absent.

```bash
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m pytest pinnde_eval/tests flow_matching/tests -q
OPENBLAS_NUM_THREADS=1 python -m flow_matching.demo_calo --features official \
    --device cuda --hidden 1024 --depth 8 --steps 100000 --clip-grad 1.0 --atom-snap
```

The last command is the recommended configuration and takes about three minutes
on an RTX 5090. Add `--relative-energy total` when the energy response matters:
it costs a little chi2 and puts the sampling fraction on the floor.

`OPENBLAS_NUM_THREADS=1` is not optional. Without it the classifier busy-waits
and a nine-second evaluation looks like a thirty-minute hang.

## 6. What comes next

In order of expected value:

1. **Explain the rank-Gaussian result.** It is the only measurement here whose
   cause is unknown, and it decides which metric the project should be
   optimising. The per-family damage is concentrated in sparsity and the
   centres, which is where to look first.
2. **Bound the voxel model's output**, by generating layer energies first and
   then the normalized pattern within each layer, which is bounded by
   construction. This is the path to numbers directly comparable with published
   submissions, which generate voxels rather than summaries.
3. **Compare against published CaloChallenge submissions** (Zenodo 15962050),
   now that our numbers are in their feature space, computed with their code,
   against measured floors.

`pinnde_eval/DEVLOG.md` is the full record, 30 sections, including the
measurements that contradicted things believed earlier.
`pinnde_eval/validate_claims.py` re-runs the three measurements in this report
that did not come from the normal scoring path.
