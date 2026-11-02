"""Reconstruction, EB3, GLADC/NetGe objectives, and paper-based SDM-ATII.

Training uses objective; total is the reconstruction subtotal where returned.
ATII has separate generator and critic objectives.
"""

import torch
from torch.nn import functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import scatter


def _mse_per_graph(prediction, target, graph_index, n_graphs):
    per_row = (prediction - target).square().mean(dim=-1)
    return scatter(per_row, graph_index, dim=0, dim_size=n_graphs, reduce='mean')


def reconstruction_errors(output, batch, *, edge_weight=1.0):
    """Return total/node/edge errors, each shaped (num_graphs,), in batch order.

    Average squared errors over features and nodes/edges within each jet.
    Jets without edges have zero edge error.
    Total is node MSE + edge_weight * edge MSE (default weight: 1).
    """
    node = _mse_per_graph(output['node'], batch.x, batch.batch, batch.num_graphs)
    edge = _mse_per_graph(
        output['edge'], batch.edge_attr,
        batch.batch[batch.edge_index[0]], batch.num_graphs,
    )
    return {
        'total': node + edge_weight * edge,
        'node': node,
        'edge': edge,
    }


def reconstruction_loss(output, batch, *, edge_weight=1.0):
    """Return scalar total/node/edge losses, averaging jet graphs equally."""
    errors = reconstruction_errors(output, batch, edge_weight=edge_weight)
    return {name: values.mean() for name, values in errors.items()}


def eb3_terms(latent, batch):
    """Per-jet sum of undirected latent-edge L2 lengths, divided by node count.

    Use the input graph's edges and unnormalized EB3 latents. Omit edgeless
    jets from the regularizer mean, matching the original implementation.
    """
    source, target = batch.edge_index
    pairs = torch.stack([torch.minimum(source, target), torch.maximum(source, target)], dim=1)
    pairs = torch.unique(pairs[source != target], dim=0)
    lengths = torch.linalg.vector_norm(latent[pairs[:, 0]] - latent[pairs[:, 1]], dim=1)
    edge_graph = batch.batch[pairs[:, 0]]
    sums = scatter(lengths, edge_graph, dim=0, dim_size=batch.num_graphs, reduce='sum')
    present = torch.bincount(edge_graph, minlength=batch.num_graphs) > 0
    nodes = torch.bincount(batch.batch, minlength=batch.num_graphs).to(latent)
    return sums[present] / nodes[present]


def contrastive_loss(noisy, clean):
    """GLADC: noisy anchors, clean candidates, off-diagonal negatives; T=0.2."""
    logits = F.normalize(noisy, dim=-1) @ F.normalize(clean, dim=-1).T / 0.2
    diagonal = torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    negatives = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    return (negatives - logits.diagonal()).mean()


def cycle_errors(model, output, batch):
    """EdgeConv cycle: re-encode reconstructed nodes AND edges, keeping connectivity."""
    z = output['latent']
    zhat = model.encode(output['node'], batch.edge_index, output['edge'])
    original_graph = model.contrastive_projection(global_max_pool(z, batch.batch))
    reconstructed_graph = model.contrastive_projection(global_max_pool(zhat, batch.batch))
    return dict(node_cycle=_mse_per_graph(zhat, z, batch.batch, batch.num_graphs),
                graph_cycle=(reconstructed_graph - original_graph).square().mean(-1))


def gladc_loss(model, batch, *, objective='full', edge_weight=1.0):
    """EdgeConv reconstruction plus the enabled cycle/contrast terms, each weighted 1."""
    batch = model.prepare(batch)
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    errors = reconstruction_errors(output, batch, edge_weight=edge_weight)
    losses = {name: value.mean() for name, value in errors.items()}
    objective_loss = errors['total']
    if objective in ('cycle', 'full'):
        cycle = cycle_errors(model, output, batch)
        losses.update({name: value.mean() for name, value in cycle.items()})
        objective_loss = objective_loss + cycle['node_cycle'] + cycle['graph_cycle']
    if objective in ('contrast', 'full'):
        clean = model.contrastive_projection(global_max_pool(output['latent'], batch.batch))
        losses['contrast'] = contrastive_loss(model.perturbed_graph(batch), clean)
        objective_loss = objective_loss + losses['contrast']
    return dict(losses, objective=objective_loss.mean())


def _netge_terms(output):
    """Unreduced errors: scalar training means and per-jet scores share these formulas."""
    x, adjacency, _, _ = output['inputs']
    identity = torch.eye(adjacency.shape[1], device=x.device, dtype=x.dtype)
    terms = dict(
        attribute=torch.linalg.vector_norm(output['node'] - x, dim=1),
        structure=torch.linalg.vector_norm(output['adjacency'] - (adjacency + identity), dim=1),
        node_cycle=(output['latent'] - output['reencoded']).square(),
        graph_cycle=(output['graph'] - output['reencoded_graph']).square(),
    )
    return terms


def netge_errors(output):
    """Original per-jet NetGe L2 reconstruction and MSE cycle terms (30 padded nodes)."""
    return {name: value.reshape(len(value), -1).mean(1) for name, value in _netge_terms(output).items()}


def netge_loss(output):
    """Original NetGe loss: objective includes all terms; total is reconstruction only."""
    terms = {name: value.mean() for name, value in _netge_terms(output).items()}
    noisy, clean = output['noisy_graph'], output['graph']
    similarity = torch.einsum('ik,jk->ij', noisy, clean)
    similarity = similarity / torch.einsum('i,j->ij', noisy.norm(dim=1), clean.norm(dim=1))
    similarity = torch.exp(similarity / 0.2)
    positive = similarity.diagonal()
    terms['contrast'] = -(positive / (similarity.sum(dim=1) - positive)).log().mean()
    reconstruction = terms['attribute'] + terms['structure']
    objective = reconstruction + terms['node_cycle'] + terms['graph_cycle'] + terms['contrast']
    return dict(terms, total=reconstruction, objective=objective)


def sdm_atii_loss(output, *, discrepancy_weight=1., kl_weight=1.):
    """Eq. (15): two separately minimized objectives, not a joint loss.

    Sum node/pair/latent entries per jet, then average jets. Adjacency BCE
    includes both edge directions and non-edges, excluding padding/self-pairs.
    KL sums both independent Gaussian branches.
    """
    graph, generated = output['graph'], output['generated']
    x, adjacency, mask = graph['x'], graph['adjacency'], graph['mask']
    node_discrepancy = ((generated['node'] - x).square() * mask[..., None]).sum((1, 2)).mean()
    adjacency_bce = F.binary_cross_entropy_with_logits(
        generated['adjacency_logits'], adjacency, reduction='none',
    )
    adjacency_discrepancy = (adjacency_bce * graph['pair_mask']).sum((1, 2)).mean()
    discrepancy = node_discrepancy + adjacency_discrepancy

    kl = x.new_zeros(())
    for mean, log_std in generated['posteriors']:
        entries = .5 * (mean.square() + (2 * log_std).exp() - 1 - 2 * log_std)
        kl = kl + (entries * mask[..., None]).sum((1, 2)).mean()

    real_critic = output['real_critic'].mean()
    generated_critic = output['generated_critic'].mean()
    generator_adversarial = -generated_critic
    generator_objective = generator_adversarial + discrepancy_weight * discrepancy + kl_weight * kl
    return dict(node_discrepancy=node_discrepancy, adjacency_discrepancy=adjacency_discrepancy,
                discrepancy=discrepancy, kl=kl, generator_adversarial=generator_adversarial,
                generator_objective=generator_objective,
                discriminator_objective=generated_critic - real_critic,
                real_critic=real_critic, generated_critic=generated_critic)


def sdm_energy_loss(output, *, discrepancy_weight=1., kl_weight=.01, margin=.2, negative_weight=.1):
    """EBGAN energy adaptation: real reconstruction plus a generated-sample hinge."""
    node_discrepancy, edge_discrepancy = output['node_discrepancy'], output['edge_discrepancy']
    discrepancy = node_discrepancy + edge_discrepancy
    real_energy = output['real_energy'].mean()
    generated_energy = output['generated_energy'].mean()
    hinge = (margin - output['generated_energy']).relu().mean()
    generator_objective = generated_energy + discrepancy_weight * discrepancy + kl_weight * output['kl']
    return dict(node_discrepancy=node_discrepancy, edge_discrepancy=edge_discrepancy,
                discrepancy=discrepancy, kl=output['kl'], generator_adversarial=generated_energy,
                generator_objective=generator_objective,
                discriminator_objective=real_energy + negative_weight * hinge,
                real_energy=real_energy, generated_energy=generated_energy, hinge=hinge)
