"""Train and validate jet models, weighting both event jets equally."""

from contextlib import nullcontext

import torch

from .dataset import combine_jets as _combine_jets
from .losses import eb3_terms, gladc_loss, netge_loss, reconstruction_loss, sdm_atii_loss, sdm_energy_loss
from .models import GLADC, NetGe, NetGeJet, SDMATII, SDMEnergy, SDMATI, SDMNAT
from .models.netge import dense_nodes, log_fractions


def _batch_losses(model, batch, device, edge_weight, reg_weight=0.0, objective='full'):
    """Return scalar losses; adversarial models keep G and D objectives separate."""
    if isinstance(model, (SDMATI, SDMNAT)):
        jets = _combine_jets([jet.to(device) for jet in batch['jets']],
                             positions=isinstance(model, SDMATI) and model.dense)
        return model(jets)
    if isinstance(model, SDMEnergy):
        jets = _combine_jets([jet.to(device) for jet in batch['jets']])
        output = model(jets)
        losses = sdm_energy_loss(output, discrepancy_weight=model.discrepancy_weight,
                                 kl_weight=model.kl_weight, margin=model.margin,
                                 negative_weight=model.negative_weight)
        # Validation mirrors the two samples used for the G and D updates.
        generated, _ = model.generator(output['graph'])
        generated_energy = model.energy(generated)
        hinge = (model.margin - generated_energy).relu().mean()
        return dict(losses, generated_energy=generated_energy.mean(), hinge=hinge,
                    discriminator_objective=losses['real_energy'] + model.negative_weight * hinge)
    if isinstance(model, (NetGe, NetGeJet, GLADC, SDMATII)):
        jets = _combine_jets([jet.to(device) for jet in batch['jets']])
        if isinstance(model, SDMATII):
            return sdm_atii_loss(model(jets), discrepancy_weight=model.discrepancy_weight,
                                 kl_weight=model.kl_weight)
        if isinstance(model, NetGeJet):
            use_cycle = objective in ('cycle', 'full')
            use_contrast = objective in ('contrast', 'full')
            output = model.reconstruct(jets, contrast=use_contrast)
            return model.objective(output, cycle_weight=float(use_cycle),
                                   contrast_weight=float(use_contrast))
        if isinstance(model, NetGe):
            return netge_loss(model.reconstruct(jets, contrast=True))
        return gladc_loss(model, jets, objective=objective, edge_weight=edge_weight)
    parts = []
    regularizers = []
    for jets in batch['jets']:
        jets = jets.to(device)
        output = model(jets.x, jets.edge_index, jets.edge_attr)
        parts.append(reconstruction_loss(output, jets, edge_weight=edge_weight))
        if reg_weight:
            regularizers.append(eb3_terms(output['latent'], jets))
    leading, subleading = parts
    losses = {name: (leading[name] + subleading[name]) / 2 for name in leading}
    losses['objective'] = losses['total']
    if reg_weight:
        terms = torch.cat(regularizers)
        losses['reg'] = terms.mean() if terms.numel() else losses['total'].new_zeros(())
        losses['objective'] = losses['total'] + reg_weight * losses['reg']
    return losses


def _train_atii_batch(model, batch, optimizers, device):
    """Algorithm 2: one generator step, then one critic step and weight clipping."""
    jets = _combine_jets([jet.to(device) for jet in batch['jets']])
    generator_optimizer = optimizers['generator']
    discriminator_optimizer = optimizers['discriminator']
    generator_optimizer.zero_grad(set_to_none=True)
    discriminator_optimizer.zero_grad(set_to_none=True)

    # Frozen critic still propagates gradients through generated nodes/adjacency.
    model.critic.requires_grad_(False)
    output = model(jets)
    losses = sdm_atii_loss(output, discrepancy_weight=model.discrepancy_weight,
                          kl_weight=model.kl_weight)
    losses['generator_objective'].backward()
    generator_optimizer.step()
    generator_optimizer.zero_grad(set_to_none=True)

    # Fresh sample from the updated generator, detached for the critic update.
    model.critic.requires_grad_(True)
    graph = output['graph']
    with torch.no_grad():
        generated = model.generator(graph)
    real_critic = model.critic(graph['x'], graph['adjacency'], graph['mask']).mean()
    generated_critic = model.critic(generated['node'], generated['adjacency'], graph['mask']).mean()
    discriminator_objective = generated_critic - real_critic
    discriminator_objective.backward()
    discriminator_optimizer.step()
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.clamp_(-model.critic_clip, model.critic_clip)
    return dict(losses, discriminator_objective=discriminator_objective,
                real_critic=real_critic, generated_critic=generated_critic)


def _train_energy_batch(model, batch, optimizers, device):
    """Generate a nearby graph, then train the AE critic on real/fresh generated graphs."""
    jets = _combine_jets([jet.to(device) for jet in batch['jets']])
    generator_optimizer = optimizers['generator']
    discriminator_optimizer = optimizers['discriminator']
    generator_optimizer.zero_grad(set_to_none=True)
    discriminator_optimizer.zero_grad(set_to_none=True)
    model.critic.requires_grad_(False)
    output = model(jets)
    losses = sdm_energy_loss(output, discrepancy_weight=model.discrepancy_weight,
                             kl_weight=model.kl_weight, margin=model.margin,
                             negative_weight=model.negative_weight)
    losses['generator_objective'].backward()
    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.)
    generator_optimizer.step()
    generator_optimizer.zero_grad(set_to_none=True)

    model.critic.requires_grad_(True)
    with torch.no_grad():
        generated, _ = model.generator(output['graph'])
    real_energy = model.energy(output['graph']).mean()
    generated_energy = model.energy(generated)
    hinge = (model.margin - generated_energy).relu().mean()
    discriminator_objective = real_energy + model.negative_weight * hinge
    discriminator_objective.backward()
    torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.)
    discriminator_optimizer.step()
    return dict(losses, discriminator_objective=discriminator_objective,
                real_energy=real_energy, generated_energy=generated_energy.mean(), hinge=hinge)


def train_epoch(model, loader, optimizer, device, *, edge_weight=1.0, reg_weight=0.0,
                scheduler=None, objective='full', grad_clip_norm=1.0):
    """Train one epoch with optional gradient clipping and per-batch OneCycleLR."""
    model.train()
    totals = {}
    n_events = 0
    for batch in loader:
        if isinstance(model, SDMATI):
            jets = _combine_jets([jet.to(device) for jet in batch['jets']], positions=model.dense)
            losses = model.train_batch(model.graph(jets), optimizer)
        elif isinstance(model, SDMATII):
            losses = _train_atii_batch(model, batch, optimizer, device)
        elif isinstance(model, SDMEnergy):
            losses = _train_energy_batch(model, batch, optimizer, device)
        else:
            optimizer.zero_grad(set_to_none=True)
            losses = _batch_losses(model, batch, device, edge_weight, reg_weight, objective)
            losses['objective'].backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        # Keep the final scheduled LR; do not advance past the last update.
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and scheduler.last_epoch + 1 < scheduler.total_steps:
            scheduler.step()
        count = batch['jets'][0].num_graphs
        n_events += count
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.) + value.detach() * count
    return {name: float(value / n_events) for name, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device, *, edge_weight=1.0, reg_weight=0.0, objective='full'):
    """Measure loss without updates; weight each event equally, including the final batch."""
    model.eval()
    totals = {}
    n_events = 0
    preserve_rng = isinstance(model, (NetGe, NetGeJet, GLADC, SDMATII, SDMEnergy, SDMATI, SDMNAT))
    device = torch.device(device)
    cuda_devices = []
    if device.type == 'cuda':
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    # Validation perturbations and loader iteration must not advance training RNG.
    rng_context = torch.random.fork_rng(devices=cuda_devices) if preserve_rng else nullcontext()
    with rng_context:
        if preserve_rng:
            validation_seed = 42 if isinstance(model, (SDMEnergy, SDMATI, SDMNAT)) else 246
            torch.random.default_generator.manual_seed(validation_seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(validation_seed)
        for batch in loader:
            losses = _batch_losses(model, batch, device, edge_weight, reg_weight, objective)
            count = batch['jets'][0].num_graphs
            n_events += count
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.) + value * count
    return {name: float(value / n_events) for name, value in totals.items()}


@torch.no_grad()
def fit_feature_scaling(model, loader):
    """Fit node/edge means and population stds on training jets; save as buffers."""
    count = dict(node=0, edge=0)
    total = dict(node=torch.zeros(1, dtype=torch.float64), edge=torch.zeros(3, dtype=torch.float64))
    squares = {name: value.clone() for name, value in total.items()}
    for batch in loader:
        for jets in batch['jets']:
            nodes = jets.x
            if model.variant == 'fraction':
                x, mask = dense_nodes(jets)
                nodes = log_fractions(x, mask)[mask]
            values = dict(node=nodes.double(), edge=jets.edge_attr.double())
            for name, value in values.items():
                count[name] += len(value)
                total[name] += value.sum(0)
                squares[name] += value.square().sum(0)
    for name in total:
        mean = total[name] / count[name]
        scale = (squares[name] / count[name] - mean.square()).sqrt()
        getattr(model, name + '_center').copy_(mean)
        getattr(model, name + '_scale').copy_(scale)
