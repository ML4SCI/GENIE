# Models

Four approaches to the same problem, in the order they were built. Each
directory holds the entrypoint, its configs, and a README stating what it is,
what it scored, and why it was superseded.

| dir | approach | cls-low AUC | status |
|---|---|---:|---|
| `0_original_flowmap/` | global flow-map PINNDE | — | **abandoned.** Does not scale past ~3 dimensions |
| `1_latent_diffusion/` | diffusion in a 96-number AE latent | 0.991 | **at its ceiling.** The decoder is the limit |
| `2_direct_pinn/` | reverse-time ODE over all 6480 voxels | 0.683 | superseded, but a strong simple baseline |
| `3_two_stage/` | energy ratios + per-layer shape | **0.578** | **current best** |

0.5 is indistinguishable from Geant4; 1.0 is trivially separable. For context,
the best published Dataset-2 submission (CaloDREAM) scored 0.531 and the
second (CaloDiffusion) 0.577.

All four share the library in `src/calods2/`. The two PINN models and the
latent model use the *same* `VelocityNet` and the *same* Heun sampler; only
the variable they operate on differs. That is the point of the layout: the
difference between the models is a choice of coordinates, not of machinery.

See `docs/MODEL_EVOLUTION.md` for the argument connecting them, and
`docs/REPLICATION.md` for full specifications.
