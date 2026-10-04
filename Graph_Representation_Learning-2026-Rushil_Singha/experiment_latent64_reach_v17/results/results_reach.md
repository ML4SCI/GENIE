## Experiment 3 — reach: does seeing further help?

Latent 64, StatsPool, identical decoder and training. Mean ± standard deviation over seeds. Lower is better.

| model | reach | seeds | params | ECF(2, β=1) | EEC wide (ΔR>0.2) | **primary (mean)** | ptD (control) | EMD | girth | jet mass | ECF(2, β=2) | support (mean) | count MAE | train loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphSAGE | 2 hops | 3 | 967,236 | 0.1683 ± 0.0011 | 0.1402 ± 0.0028 | 0.1542 ± 0.0019 | 0.0168 ± 0.0080 | 0.0179 ± 0.0007 | 0.1399 ± 0.0043 | 0.1644 ± 0.0061 | 0.1476 ± 0.0055 | 0.1506 ± 0.0051 | 0.0161 ± 0.0208 | 0.0161 ± 0.0023 |
| ChebNet K=2 | 2 hops | 3 | 967,236 | 0.1647 ± 0.0023 | 0.1366 ± 0.0052 | 0.1506 ± 0.0037 | 0.0117 ± 0.0004 | 0.0173 ± 0.0004 | 0.1337 ± 0.0035 | 0.1579 ± 0.0036 | 0.1425 ± 0.0046 | 0.1447 ± 0.0038 | 0.0003 ± 0.0000 | 0.0146 ± 0.0002 |
| ChebNet K=6 | 10 hops | 3 | 1,491,524 | 0.1740 ± 0.0030 | 0.1422 ± 0.0020 | 0.1581 ± 0.0018 | 0.0129 ± 0.0010 | 0.0177 ± 0.0001 | 0.1417 ± 0.0022 | 0.1673 ± 0.0018 | 0.1519 ± 0.0017 | 0.1536 ± 0.0019 | 0.0067 ± 0.0074 | 0.0150 ± 0.0006 |

### The comparisons that matter

Difference in paired relative error, **positive = the second model is better**. p from a Welch t-test across seeds — with 3 seeds per model, treat p < 0.05 as suggestive, not conclusive.

| comparison | isolates | primary Δ (p) | support Δ (p) | ptD control Δ (p) | reading |
|---|---|---|---|---|---|
| GraphSAGE → ChebNet K=2 | the operator (reach held at 2 hops) | +0.0036 (0.231) | +0.0059 (0.187) | +0.0051 (0.385) | no separation beyond seed noise |
| ChebNet K=2 → ChebNet K=6 | **reach** (operator held fixed) | -0.0074 (0.054) | -0.0089 (0.037) | -0.0012 (0.166) | no separation beyond seed noise |
| GraphSAGE → ChebNet K=6 | operator + reach together | -0.0039 (0.066) | -0.0030 (0.419) | +0.0039 (0.484) | no separation beyond seed noise |

**How to read this.** A reach effect should show up in the *primary* column and **not** in the support column (those observables need no reach) or the control column (no geometry at all). If all three move together, the models differ in how well they trained, not in what they capture.

### Split by jet size

A compression-driven effect should concentrate in jets carrying more numbers than the latent holds (> 21 particles at latent 64).

| model | primary, small jets | primary, large jets |
|---|---|---|
| GraphSAGE | 0.4276 ± 0.0123 | 0.1596 ± 0.0021 |
| ChebNet K=2 | 0.4321 ± 0.0032 | 0.1557 ± 0.0039 |
| ChebNet K=6 | 0.4256 ± 0.0484 | 0.1636 ± 0.0019 |

### Per jet type (primary mean)

Top jets carry three-prong structure — where reach should matter most, if anywhere. Each column is normalised by that jet type's own spread, so **compare models down a column, never across columns**.

| model | gluon | quark | top |
|---|---|---|---|
| GraphSAGE | 0.1743 ± 0.0023 | 0.1769 ± 0.0019 | 0.2883 ± 0.0050 |
| ChebNet K=2 | 0.1714 ± 0.0033 | 0.1724 ± 0.0017 | 0.2806 ± 0.0108 |
| ChebNet K=6 | 0.1771 ± 0.0024 | 0.1766 ± 0.0040 | 0.3006 ± 0.0059 |

