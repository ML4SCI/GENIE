"""Physical-feature NAT: jointly train a generator and normal/generated classifier."""

import torch
from torch import nn
from torch.nn import functional as F

from .netge import NetGeMaxBilinear, _JetEncoder
from .sdm_energy import SDMEnergy, FeatureGenerator, _encode, _decode, feature_errors, _posterior_kl


NAT_VARIANTS = ('energy', 'message', 'paired', 'control')


class MessageClassifier(nn.Module):
    """Max-message encoder, mean/max pooling and a normal-class logit."""

    def __init__(self):
        super().__init__()
        self.encoder = _JetEncoder(16)
        self.encoder.aggr = 'max'
        self.encoder.proj_head = nn.Identity()
        self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))

    def representations(self, graph):
        mask = graph['mask']
        z, _ = self.encoder(graph['x'].reshape(*mask.shape, 1),
                            graph['edge_index'], graph['edge'], mask)
        mean = (z * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        maximum = z.masked_fill(~mask[..., None], -torch.inf).max(1).values
        return torch.cat([mean, maximum], -1)

    def forward(self, graph):
        return self.head(self.representations(graph)).squeeze(-1)


class EnergyClassifier(nn.Module):
    """Normal logit = learned intercept − AE reconstruction error."""

    def __init__(self):
        super().__init__()
        self.ae = NetGeMaxBilinear()
        self.intercept = nn.Parameter(torch.zeros(()))

    def energy(self, graph):
        nodes, edges = _decode(self.ae, _encode(self.ae, graph), graph)
        node_error, edge_error = feature_errors(graph, nodes, edges)
        return node_error + edge_error

    def forward(self, graph):
        return self.intercept - self.energy(graph)


class PairClassifier(nn.Module):
    """One event logit, symmetric in the two jets; jet slots occupy matching halves."""

    def __init__(self, single):
        super().__init__()
        self.single = single
        self.relationship = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.relationship[-1].weight)
        nn.init.zeros_(self.relationship[-1].bias)

    def forward(self, graph):
        hidden = self.single.representations(graph)
        first, second = hidden.chunk(2)
        logit0, logit1 = self.single.head(hidden).squeeze(-1).chunk(2)
        symmetric = torch.cat([.5 * (first + second), (first - second).abs()], -1)
        return .5 * (logit0 + logit1) + self.relationship(symmetric).squeeze(-1)


class SDMNAT(nn.Module):
    """Fixed Unique-6 features; one optimizer for both independent networks.

    Input: a PyG jet batch, leading jets followed by matching subleading jets.
    Forward returns scalar losses. event_scores returns one score per event.
    """

    prepare = SDMEnergy.prepare
    graph = SDMEnergy.graph

    def __init__(self, variant='energy', max_subjets=30):
        super().__init__()
        self.variant = variant
        self.max_subjets = max_subjets
        for name, size in (('node', 1), ('edge', 3)):
            self.register_buffer(name + '_center', torch.zeros(size))
            self.register_buffer(name + '_scale', torch.ones(size))
        self.generator = FeatureGenerator('netge')
        # Match the experiment's generator initialization and sampling stream.
        with torch.random.fork_rng(devices=[]):
            self.classifier = (MessageClassifier() if variant in ('message', 'paired')
                               else EnergyClassifier())
        if variant == 'paired':
            with torch.random.fork_rng(devices=[]):
                self.classifier = PairClassifier(self.classifier)
        if variant == 'control':
            self.generator.requires_grad_(False)
            self.classifier.intercept.requires_grad_(False)

    def losses(self, graph):
        if self.variant == 'control':
            objective = self.classifier.energy(graph).mean()
            return dict(reconstruction=objective, objective=objective)
        generated, penalties = self.generator(graph)
        real_bce = F.softplus(-self.classifier(graph)).mean()
        generated_bce = F.softplus(self.classifier(generated)).mean()
        discrepancy = penalties['node_discrepancy'] + penalties['edge_discrepancy']
        objective = real_bce + generated_bce + discrepancy + .01 * penalties['kl']
        return dict(penalties, real_bce=real_bce, generated_bce=generated_bce, objective=objective)

    def forward(self, jets):
        return self.losses(self.graph(jets))

    def objective_components(self, graph):
        """Deterministic objective at the posterior mean, per jet or paired event."""
        mask = graph['mask']
        mean, log_std = self.generator.posterior(graph)
        nodes, edges = _decode(self.generator.ae, mean * mask[..., None], graph)
        nodes = nodes * mask.flatten()[:, None]
        node_error, edge_error = feature_errors(graph, nodes, edges)
        generated = dict(graph, x=nodes, edge=edges)
        terms = dict(node_discrepancy=node_error, edge_discrepancy=edge_error,
                     weighted_kl=.01 * _posterior_kl(mean, log_std, mask))
        if self.variant == 'paired':
            terms = {name: .5 * sum(value.chunk(2)) for name, value in terms.items()}
        return dict(terms, real_bce=F.softplus(-self.classifier(graph)),
                    generated_bce=F.softplus(self.classifier(generated)))

    def event_scores(self, jets):
        graph = self.graph(jets)
        if self.variant == 'control':
            return dict(reconstruction=sum(self.classifier.energy(graph).double().chunk(2)))
        classifier = torch.sigmoid(-self.classifier(graph).double())
        objective = sum(value.double() for value in self.objective_components(graph).values())
        if self.variant != 'paired':
            classifier, objective = sum(classifier.chunk(2)), sum(objective.chunk(2))
        return dict(classifier=classifier, full_objective=objective)
