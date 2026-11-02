"""Soft Drop prediction targets from the 30-subjet graphs (beta=0)."""

import fastjet as fj
import numpy as np


ZCUT = 0.1
RECLUSTER_RADIUS = 1000.0  # Keep all input subjets in one C/A tree.


def jet_softdrop(pt, positions):
    """Mass retention, scalar-pT retention, splitting fraction zg and opening angle Rg."""
    pt = np.asarray(pt, np.float64)
    positions = np.asarray(positions, np.float64)
    y, phi = positions.T
    # Massless four-vectors in FastJet order: px, py, pz, E.
    vectors = np.column_stack([
        pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(y), pt*np.cosh(y),
    ])
    particles = [fj.PseudoJet(*vector) for vector in vectors]
    # Preserve indices for the retained scalar-pT sum.
    for index, particle in enumerate(particles):
        particle.set_user_index(index)

    jet_definition = fj.JetDefinition(fj.cambridge_algorithm, RECLUSTER_RADIUS)
    sequence = fj.ClusterSequence(particles, jet_definition)
    original_jet = sequence.inclusive_jets()[0]
    groomed_jet = original_jet
    zg = rg = 0.
    # Undo C/A merges, dropping the softer branch until z > ZCUT.
    while True:
        harder, softer = fj.PseudoJet(), fj.PseudoJet()
        if not groomed_jet.has_parents(harder, softer):
            break
        if softer.pt() > harder.pt():
            harder, softer = softer, harder
        z = softer.pt() / (harder.pt() + softer.pt())
        if z > ZCUT:
            zg, rg = z, harder.delta_R(softer)
            break
        groomed_jet = harder

    kept = np.array(sorted(p.user_index() for p in groomed_jet.constituents()))
    groomed_mass2 = max(groomed_jet.m2(), 0.) if len(kept) > 1 else 0.
    original_mass2 = max(original_jet.m2(), 1e-24)
    mass_ratio = np.sqrt(groomed_mass2 / original_mass2) if len(pt) > 1 else 0.
    pt_ratio = pt[kept].sum() / pt.sum()
    return np.array([mass_ratio, pt_ratio, zg, rg])


def event_softdrop(data):
    """Eight targets: sums and absolute differences of the two jets' four observables."""
    pt, positions, mask = [data[name].numpy() for name in ('pt', 'positions', 'mask')]
    jet_features = np.empty((len(pt), 2, 4), dtype=np.float64)
    for event in range(len(pt)):
        for slot in (0, 1):
            selected = mask[event, slot]
            jet_features[event, slot] = jet_softdrop(
                pt[event, slot, selected], positions[event, slot, selected],
            )
    # Keep event features unchanged when the two jets are swapped.
    first_jet, second_jet = jet_features[:, 0], jet_features[:, 1]
    feature_sum = first_jet + second_jet
    feature_difference = np.abs(first_jet - second_jet)
    return np.concatenate([feature_sum, feature_difference], axis=1)
