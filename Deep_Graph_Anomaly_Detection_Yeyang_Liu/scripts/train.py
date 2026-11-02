"""Train a jet model on the reference sample (CWoLa SB) by default; save epochs and metrics."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import tables
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.checkpoint import save_checkpoint
from src.dataset import EventDataset
from src.metrics import score_metrics
from src.models import create_model
from src.models.gladc import GLADC_VARIANTS
from src.models.netge import NETGE_VARIANTS
from src.models.sdm_ati import ATI_VARIANTS
from src.models.sdm_nat import NAT_VARIANTS
from src.scoring import score_loader, scoring_options
from src.training import fit_feature_scaling, train_epoch, validate


def _model_defaults(model, variant=None):
    """Default training settings for each model."""
    variant = variant or ('energy' if model in ('sdm_ati', 'sdm_nat') else 'original')
    values = dict(model=model, sample='reference', seed=42, epochs=50, eval_interval=5,
                  batch_size=256, lr=.003, edge_weight=1., reg_weight=0.,
                  objective='reconstruction', score=scoring_options(model, variant=variant)[0],
                  graph_dir=Path('data/graphs'), device='cpu', num_workers=0)
    if model == 'gladc':
        values.update(gladc_variant='original', objective='full')
    elif model == 'netge':
        values['netge_variant'] = variant
        if variant in ('original', 'jet'):
            values.update(batch_size=150, lr=.0001)
        if variant == 'original':
            values['objective'] = 'full'
    elif model == 'sdm_atii':
        values.update(batch_size=128, lr=.001, objective='atii',
                      sdm_discrepancy_weight=1., sdm_kl_weight=1., sdm_critic_clip=.01)
    elif model == 'sdm_energy':
        values.update(lr=.0002, objective='energy', sdm_backbone='netge',
                      sdm_margin=.2, sdm_negative_weight=.1, sdm_critic_context=False,
                      sdm_discrepancy_weight=1., sdm_kl_weight=.01)
    elif model == 'sdm_ati':
        values.update(batch_size=128, lr=.001, objective='ati', sdm_variant=variant)
    elif model == 'sdm_nat':
        values.update(lr=.0002, objective='nat', sdm_variant=variant)
    return values


def _resume_arguments(options, parser):
    checkpoint = torch.load(options['resume'], map_location='cpu', weights_only=True)
    required = {'model', 'settings', 'epoch', 'optimizer', 'scheduler', 'rng', 'history'}
    if not required.issubset(checkpoint):
        parser.error('Resume requires a training checkpoint with optimizer and RNG state')
    saved = checkpoint['settings']
    for name, value in options.items():
        if name in ('resume', 'device', 'num_workers'):
            continue
        value = str(value.resolve()) if isinstance(value, Path) else value
        if value != saved[name]:
            parser.error(f'--{name.replace("_", "-")} cannot change on resume')
    values = {**saved, 'resume': options['resume'], 'output_dir': options['resume'].resolve().parent,
              'device': options.get('device', saved['device']),
              'num_workers': options.get('num_workers', saved['num_workers'])}
    return values, checkpoint


def _check_options(args, parser):
    if args.epochs < 1 or args.eval_interval < 1:
        parser.error('--epochs and --eval-interval must be at least 1')
    if args.reg_weight < 0 or (args.reg_weight and args.model != 'edgeconv'):
        parser.error('--reg-weight is a nonnegative EdgeConv EB3 weight')
    if hasattr(args, 'netge_variant') and args.model != 'netge':
        parser.error('--netge-variant applies only to NetGe')
    if hasattr(args, 'gladc_variant') and args.model != 'gladc':
        parser.error('--gladc-variant applies only to GLADC')
    if hasattr(args, 'sdm_critic_context') and args.model != 'sdm_energy':
        parser.error('--sdm-critic-context applies only to SDM Energy')
    if hasattr(args, 'sdm_variant'):
        variants = {'sdm_ati': ATI_VARIANTS, 'sdm_nat': NAT_VARIANTS}.get(args.model, ())
        if args.sdm_variant not in variants:
            parser.error(f'--sdm-variant for {args.model}: {", ".join(variants)}')
    auxiliary = ('reconstruction', 'cycle', 'contrast', 'full')
    allowed = dict(edgeconv=('reconstruction',), gladc=auxiliary,
                   sdm_atii=('atii',), sdm_energy=('energy',), sdm_ati=('ati',), sdm_nat=('nat',))
    if args.model == 'netge':
        allowed['netge'] = {'original': ('full',), 'fraction': ('reconstruction',),
                            'jet': auxiliary, 'max_bilinear': auxiliary}[args.netge_variant]
    if args.objective not in allowed[args.model]:
        parser.error(f'{args.model} objectives: {", ".join(allowed[args.model])}')
    if args.model in ('netge', 'sdm_atii', 'sdm_energy', 'sdm_ati', 'sdm_nat') and args.edge_weight != 1:
        parser.error(f'{args.model} uses unit edge weighting')
    variant = getattr(args, 'netge_variant', getattr(args, 'sdm_variant', 'original'))
    default, available = scoring_options(args.model, variant=variant)
    if args.score not in available:
        parser.error(f'--score for {args.model}: {", ".join(available)}')
    if args.score == 'total':
        args.score = default


def _arguments():
    # Unspecified training options come from the checkpoint when resuming.
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument('--resume', type=Path, default=None, help='continue an interrupted last.pt run')
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--graph-dir', type=Path)
    parser.add_argument('--sample', choices=['SR', 'reference'],
                        help='reference (default) = CWoLa sideband or IAD extra QCD')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model', choices=['edgeconv', 'gladc', 'netge', 'sdm_atii', 'sdm_energy', 'sdm_ati', 'sdm_nat'])
    parser.add_argument('--netge-variant', choices=NETGE_VARIANTS)
    parser.add_argument('--gladc-variant', choices=GLADC_VARIANTS)
    parser.add_argument('--objective', choices=['reconstruction', 'cycle', 'contrast', 'full'],
                        help='GLADC/NetGe reconstruction and auxiliary tasks')
    parser.add_argument('--score', help='primary anomaly score; default depends on model/variant')
    parser.add_argument('--epochs', type=int, help='total epochs, default 50')
    parser.add_argument('--batch-size', type=int, help='events per batch, two jets each; defaults follow the model recipe')
    parser.add_argument('--lr', type=float, help='fixed learning rate, or peak for OneCycleLR')
    parser.add_argument('--edge-weight', type=float)
    parser.add_argument('--reg-weight', type=float,
                        help='EB3 regularization weight; 0 = baseline, 1 = previous EB3 setting')
    parser.add_argument('--sdm-critic-context', action='store_true', help='add mean incident-edge context to the NetGe critic')
    parser.add_argument('--sdm-variant', help='ATI: original, energy, dense, control, dense_control; NAT: energy, message, paired, control')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-interval', type=int, help='test AUC/SIC interval, default 5; also evaluate final epoch')
    parser.add_argument('--device', help='cpu or cuda:0, cuda:1, ...; default cpu for new runs')
    parser.add_argument('--num-workers', type=int)
    options = vars(parser.parse_args())
    checkpoint = None
    if options['resume']:
        values, checkpoint = _resume_arguments(options, parser)
    else:
        for name in ('split_file', 'output_dir'):
            if name not in options:
                parser.error(f'--{name.replace("_", "-")} is required for a new run')
        variant = options.get('netge_variant', options.get('sdm_variant'))
        values = _model_defaults(options.get('model', 'edgeconv'), variant)
        values.update(options)
    for name in ('split_file', 'graph_dir', 'output_dir'):
        values[name] = Path(values[name])
    args = argparse.Namespace(**values)
    _check_options(args, parser)
    return args, checkpoint


def _write_history(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_optimizer(model, args, total_steps):
    if args.model == 'sdm_nat':
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.01 * args.lr)
        return optimizer, scheduler, dict(name='CosineAnnealingLR', T_max=args.epochs, eta_min=.01 * args.lr), 1.
    if args.model == 'sdm_ati':
        modules = dict(discriminator=model.critic)
        if model.generator is not None:
            modules['generator'] = model.generator
        optimizer_type = torch.optim.RMSprop if args.sdm_variant == 'original' else torch.optim.Adam
        optimizer = {name: optimizer_type(module.parameters(), lr=args.lr) for name, module in modules.items()}
        schedule = (None if args.sdm_variant == 'original' else
                    dict(name='epoch_cosine', initial_lr=args.lr, final_lr=.01 * args.lr))
        return optimizer, None, schedule, None
    if args.model == 'sdm_energy':
        optimizer = {name: torch.optim.Adam(module.parameters(), lr=args.lr, betas=(0., .9))
                     for name, module in (('generator', model.generator), ('discriminator', model.critic))}
        schedule = dict(name='epoch_cosine', initial_lr=args.lr, final_lr=.01 * args.lr)
        return optimizer, None, schedule, 1.
    if args.model == 'sdm_atii':
        optimizer = {name: torch.optim.RMSprop(module.parameters(), lr=args.lr)
                     for name, module in (('generator', model.generator), ('discriminator', model.critic))}
        return optimizer, None, None, None
    if args.model == 'netge' and args.netge_variant in ('original', 'jet'):
        return torch.optim.Adam(model.parameters(), lr=args.lr), None, None, None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    pct_start = min(max(max(2, int(.02 * total_steps)) / total_steps, .01), .9)
    schedule = dict(max_lr=args.lr, total_steps=total_steps, pct_start=pct_start,
                    anneal_strategy='linear', div_factor=5., final_div_factor=3.)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **schedule)
    return optimizer, scheduler, dict(name='OneCycleLR', **schedule), 1.


def _restore_training(checkpoint, model, optimizer, scheduler, loader, args):
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and scheduler.total_steps != checkpoint['scheduler']['total_steps']:
        raise ValueError('Training loader length changed; cannot resume the saved OneCycleLR schedule')
    model.load_state_dict(checkpoint['model'])
    if isinstance(optimizer, dict):
        for name, part in optimizer.items():
            part.load_state_dict(checkpoint['optimizer'][name])
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['rng']['torch'])
    loader.generator.set_state(checkpoint['rng']['loader'])
    if checkpoint['rng']['cuda'] is not None and torch.device(args.device).type == 'cuda':
        torch.cuda.set_rng_state(checkpoint['rng']['cuda'], device=args.device)
    history = checkpoint['history']
    for name, rows in history.items():
        if rows:
            _write_history(args.output_dir / f'{name}.csv', rows)
    print(f'Resuming after epoch {checkpoint["epoch"]}', flush=True)
    return history, checkpoint['epoch'] + 1


def _evaluate(model, loader, args, epoch):
    scored = score_loader(model, loader, args.device, edge_weight=args.edge_weight, score=args.score)
    metrics = score_metrics(scored['total'], scored['truth'])
    if args.model == 'edgeconv':
        return metrics, [dict(epoch=epoch, split='test', **metrics)]
    variant = getattr(args, 'netge_variant', getattr(args, 'sdm_variant', 'original'))
    available = scoring_options(args.model, variant=variant)[1]
    rows = []
    for name in available:
        if name in scored:
            values = metrics if name == 'total' else score_metrics(scored[name], scored['truth'])
            rows.append(dict(epoch=epoch, split='test', score=name, **values))
    return metrics, rows


def main():
    args, checkpoint = _arguments()
    variant = getattr(args, 'netge_variant', 'original')
    if checkpoint is not None and checkpoint['epoch'] >= args.epochs:
        for name, rows in checkpoint['history'].items():
            if rows:
                _write_history(args.output_dir / f'{name}.csv', rows)
        print(f'Run already completed epoch {checkpoint["epoch"]}', flush=True)
        return

    datasets = []
    weak_label = {'SR': 1, 'reference': 0}[args.sample]
    for part in ('train', 'validation'):
        dataset = EventDataset(args.split_file, split=part, graph_dir=args.graph_dir)
        indices = np.flatnonzero(dataset.events['weak_label'] == weak_label)
        if len(indices) == 0:
            raise ValueError(f'No {args.sample} events in {part}')
        datasets.append(Subset(dataset, indices))
    # Keep the max/bilinear experiment's seed + epoch event permutation.
    ordered = variant == 'max_bilinear' or args.model in ('sdm_energy', 'sdm_ati', 'sdm_nat')
    epoch_order = np.arange(len(datasets[0])) if ordered else None
    training_loader = DataLoader(
        datasets[0], batch_size=args.batch_size, shuffle=epoch_order is None,
        sampler=epoch_order, num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        datasets[1], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        generator=(torch.Generator().manual_seed(args.seed)
                   if args.model in ('sdm_ati', 'sdm_nat') else None),
    )
    test_loader = DataLoader(
        EventDataset(args.split_file, split='test', graph_dir=args.graph_dir),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        # Monitoring must not consume the training RNG stream.
        generator=torch.Generator().manual_seed(args.seed),
    )

    standardized_netge = args.model == 'netge' and variant in ('fraction', 'max_bilinear')
    standardized_gladc = args.model == 'gladc' and args.gladc_variant in ('standardized', 'fraction')
    energy = args.model == 'sdm_energy'
    ati, nat = args.model == 'sdm_ati', args.model == 'sdm_nat'
    if (energy or ati or nat) and checkpoint is None:
        dataset = datasets[0].dataset
        source = dataset.events[datasets[0].indices[0]]['source']
        with tables.open_file(dataset.graph_paths[source]) as graphs:
            args.max_subjets = int(graphs.root._v_attrs.max_subjets)
        args.n_train = len(datasets[0])
    torch.manual_seed(args.seed)
    model = create_model(vars(args)).to(args.device)
    standardized = standardized_netge or standardized_gladc or energy or nat or (ati and args.sdm_variant != 'original')
    if checkpoint is None and standardized:
        scaling_loader = DataLoader(datasets[0], batch_size=256, shuffle=False,
                                    num_workers=args.num_workers,
                                    generator=torch.Generator().manual_seed(args.seed))
        fit_feature_scaling(model, scaling_loader)
        if ati:
            model.fit_reference(scaling_loader, args.device)
            if args.sdm_variant == 'dense':
                args.ati_shape_size = len(model.shape_reference)
    atii = args.model == 'sdm_atii'
    adversarial = atii or energy or ati
    plain_adam = args.model == 'netge' and not standardized_netge
    total_steps = args.epochs * len(training_loader)
    optimizer, scheduler, schedule_settings, grad_clip_norm = _make_optimizer(model, args, total_steps)
    history = {'losses': [], 'metrics': []}
    start_epoch = 1
    if checkpoint is None:
        settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != 'resume'}
        rmsprop = atii or (ati and args.sdm_variant == 'original')
        settings.update(optimizer='RMSprop' if rmsprop else ('Adam' if plain_adam or energy or ati or nat else 'AdamW'),
                        weight_decay=0. if adversarial or plain_adam or nat else 0.01,
                        grad_clip_norm=grad_clip_norm, scheduler=schedule_settings,
                        regularization='eb3' if args.reg_weight else 'none',
                        n_train=len(datasets[0]), n_validation=len(datasets[1]))
        if energy:
            settings.update(betas=[0., .9], method='ATII-conditioned VAE + EBGAN energy critic')
        if standardized:
            settings['feature_scaling'] = {name: getattr(model, name).tolist() for name in
                                          ('node_center', 'node_scale', 'edge_center', 'edge_scale')}
        args.output_dir.mkdir(parents=True)
        with (args.output_dir / 'settings.json').open('w') as file:
            json.dump(settings, file, indent=2)
    else:
        settings = checkpoint['settings']
        history, start_epoch = _restore_training(checkpoint, model, optimizer, scheduler, training_loader, args)

    print(f'{args.sample}: {len(datasets[0]):,} training / {len(datasets[1]):,} validation events', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if ati:
            model.start_epoch()
        if energy or (ati and args.sdm_variant != 'original'):
            factor = .01 + .99 * (1 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1))) / 2
            for part in optimizer.values():
                for group in part.param_groups:
                    group['lr'] = args.lr * factor
        if epoch_order is not None:
            epoch_order[:] = np.random.RandomState(args.seed + epoch).permutation(len(epoch_order))
        training = train_epoch(model, training_loader, optimizer, args.device,
                               edge_weight=args.edge_weight, reg_weight=args.reg_weight,
                               scheduler=scheduler, objective=args.objective, grad_clip_norm=grad_clip_norm)
        validation = validate(model, validation_loader, args.device,
                              edge_weight=args.edge_weight, reg_weight=args.reg_weight, objective=args.objective)
        if ati:
            model.end_epoch()
        if nat:
            scheduler.step()
        primary_optimizer = optimizer['discriminator'] if adversarial else optimizer
        row = {'epoch': epoch, 'lr': primary_optimizer.param_groups[0]['lr']}
        row.update({f'train_{name}': value for name, value in training.items()})
        row.update({f'validation_{name}': value for name, value in validation.items()})
        history['losses'].append(row)
        if adversarial:
            message = (f'Epoch {epoch}/{args.epochs}: '
                       f'train_G={training["generator_objective"]:.6g} '
                       f'train_D={training["discriminator_objective"]:.6g} '
                       f'validation_G={validation["generator_objective"]:.6g} '
                       f'validation_D={validation["discriminator_objective"]:.6g}')
        else:
            message = (f'Epoch {epoch}/{args.epochs}: train_objective={training["objective"]:.6g} '
                       f'validation_objective={validation["objective"]:.6g}')
        if args.reg_weight:
            message += f' reg={training["reg"]:.6g}'
        evaluate_now = epoch % args.eval_interval == 0 or epoch == args.epochs
        if evaluate_now:
            metrics, rows = _evaluate(model, test_loader, args, epoch)
            history['metrics'].extend(rows)
            message += f' AUC={metrics["auc"]} maxSIC(KS)={metrics["max_sic_ks"]}'

        rng = dict(torch=torch.get_rng_state(), loader=training_loader.generator.get_state(),
                   cuda=torch.cuda.get_rng_state(args.device) if torch.device(args.device).type == 'cuda' else None)
        optimizer_state = ({name: part.state_dict() for name, part in optimizer.items()}
                           if adversarial else optimizer.state_dict())
        state = {
            'model': model.state_dict(), 'optimizer': optimizer_state,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch, 'settings': settings,
            'rng': rng, 'history': history,
        }
        save_checkpoint(args.output_dir / 'last.pt', state)
        if evaluate_now and args.model != 'edgeconv':
            save_checkpoint(args.output_dir / f'epoch{epoch:02d}.pt', state)
        for name, rows in history.items():
            if rows:
                _write_history(args.output_dir / f'{name}.csv', rows)
        print(message, flush=True)


if __name__ == '__main__':
    main()
