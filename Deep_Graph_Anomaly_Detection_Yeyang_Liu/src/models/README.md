# Models

| Module | Model |
| --- | --- |
| [edgeconv.py](edgeconv.py) | EdgeConv autoencoder for node and physical-edge reconstruction. |
| [gladc.py](gladc.py) | EdgeConv–GLADC with cycle and contrastive objectives. |
| [netge.py](netge.py) | NetGe variants: original, jet, relative-pT and max-message/bilinear. |
| [physics_ae.py](physics_ae.py) | NetGe autoencoder with mass, τ, EFP and τ-ratio prediction heads. |
| [dual_ae.py](dual_ae.py) | Soft Drop NetGe and pair-attention autoencoders coupled through InfoNCE. |
| [sdm.py](sdm.py) | ATII: GIN variational graph generator and Wasserstein critic. |
| [sdm_energy.py](sdm_energy.py) | ATII with an autoencoder energy critic and optional edge context. |
| [sdm_ati.py](sdm_ati.py) | ATI: noise-based graph generation with scalar or autoencoder critics. |
| [sdm_nat.py](sdm_nat.py) | NAT: jointly trained graph generator and classifier. |
| [__init__.py](__init__.py) | Create models from configuration settings. |
