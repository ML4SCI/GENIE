"""Toy demo: train flow matching on a 2-D Gaussian-mixture and score it.

Run from the ``Tina`` folder (or paste into Colab):

    python -m flow_matching.demo

It trains a velocity field on a fixed GMM, samples from it, and reports the
``pinnde_eval`` metrics of generated-vs-true. Success looks like AUC heading
toward ~0.5 and SWD heading toward the null floor (~0.05 for this toy), i.e.
the generator is becoming statistically indistinguishable from the truth.
"""

import os
import sys

# Make pinnde_eval and flow_matching importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnde_eval import evaluate, report
from pinnde_eval.data import gmm_params, sample_gmm

from .core import sample
from .train import train_flow_matching


def main(d=2, k=6, n_data=20000, n_eval=5000, n_steps=4000, seed=0):
    params = gmm_params(d=d, k=k, seed=seed)
    data = sample_gmm(params, n_data, seed=seed + 1)        # training data
    real_eval = sample_gmm(params, n_eval, seed=seed + 3)   # held-out truth

    print(f"training flow matching: d={d}, k={k}, n_data={n_data}, steps={n_steps}")
    model, history = train_flow_matching(
        data, dim=d, n_steps=n_steps, seed=seed,
        monitor_every=max(1, n_steps // 8), monitor_real=real_eval,
    )

    print("\nstep |      loss |        mmd |        swd")
    for step, loss, m, s in history:
        print(f"{step:5d} | {loss:9.4f} | {m:10.4e} | {s:10.4f}")

    gen = sample(model, n_eval, d, steps=50, seed=seed)
    print()
    report(evaluate(real_eval, gen, tier="full", seed=seed),
           title="flow matching vs truth (2-D GMM)")
    return model, history


if __name__ == "__main__":
    main()
