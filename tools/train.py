# RegionBGC adaptation for BGC region prediction; upstream attribution is retained.
"""by lyuwenyu
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.misc import distributed, set_seed
from src.solver import TASKS


def _validate_cli_paths(args) -> None:
    path_fields = {
        'config': [args.config],
        'resume': [args.resume] if args.resume else [],
        'tuning': [args.tuning] if args.tuning else [],
        'train_data': [args.train_data] if args.train_data else [],
        'train_features': args.train_features or [],
        'val_data': [args.val_data] if args.val_data else [],
        'val_features': args.val_features or [],
    }
    for field_name, values in path_fields.items():
        for value in values:
            path = Path(value).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f'--{field_name.replace("_", "-")} does not exist: {path}')


def _build_config_overrides(args) -> dict:
    overrides = {
        'resume': args.resume,
        'use_amp': args.amp,
        'tuning': args.tuning,
        # Keep the sampler seed in the resolved configuration so the
        # single-process loader can reproduce the original one-rank
        # DistributedSampler ordering exactly.
        'seed': args.seed if args.seed is not None else 6,
    }
    if args.output_dir:
        overrides['output_dir'] = str(Path(args.output_dir).expanduser().resolve())
    if args.epochs is not None:
        overrides['epoches'] = args.epochs

    split_values = {
        'train_dataloader': (args.train_data, args.train_features),
        'val_dataloader': (args.val_data, args.val_features),
    }
    for split_name, (data_file, feature_files) in split_values.items():
        dataloader_override = {}
        dataset_override = {}
        if data_file:
            dataset_override['data_file'] = str(Path(data_file).expanduser().resolve())
        if feature_files:
            dataset_override['features_files'] = [
                str(Path(path).expanduser().resolve()) for path in feature_files
            ]
        if dataset_override:
            dataloader_override['dataset'] = dataset_override
        if args.batch_size is not None:
            dataloader_override['batch_size'] = args.batch_size
        if args.num_workers is not None:
            dataloader_override['num_workers'] = args.num_workers
            if args.num_workers == 0:
                dataloader_override['persistent_workers'] = False
                dataloader_override['prefetch_factor'] = None
        if dataloader_override:
            overrides[split_name] = dataloader_override
    eval_loader_override = {}
    if args.batch_size is not None:
        eval_loader_override['batch_size'] = args.batch_size
    if args.num_workers is not None:
        eval_loader_override['num_workers'] = args.num_workers
        if args.num_workers == 0:
            eval_loader_override['persistent_workers'] = False
            eval_loader_override['prefetch_factor'] = None
    if eval_loader_override:
        overrides['bgc_eval'] = {'dataloader': eval_loader_override}
    return overrides


def _resolve_training_paths(cfg) -> None:
    """Resolve YAML defaults against the repository, independent of shell cwd."""
    root = Path(__file__).resolve().parents[1]
    for split in ('train_dataloader', 'val_dataloader'):
        dataset = cfg.yaml_cfg[split]['dataset']
        paths = [dataset['data_file'], *dataset['features_files']]
        resolved = []
        for value in paths:
            path = Path(value).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f'{split} input does not exist: {path}')
            resolved.append(str(path))
        dataset['data_file'], dataset['features_files'] = resolved[0], resolved[1:]
    output = Path(cfg.output_dir).expanduser()
    cfg.output_dir = str((root / output).resolve() if not output.is_absolute() else output.resolve())
    cfg.yaml_cfg['output_dir'] = cfg.output_dir


def main(args, ) -> None:
    '''main
    '''
    _validate_cli_paths(args)

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != 1:
        raise RuntimeError(
            'RegionBGC training supports one process and one GPU only. '
            'Run tools/train.py directly instead of using torchrun.'
        )
    if args.gpu_id is not None:
        if args.gpu_id < 0:
            raise ValueError('--gpu-id must be non-negative')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    distributed.init_distributed()

    # Keep training reproducible while allowing an explicit CLI override.
    seed = args.seed if args.seed is not None else 6
    set_seed(seed)

    if args.tuning and args.resume:
        raise ValueError('--tuning and --resume cannot be used together')

    cfg = YAMLConfig(args.config, **_build_config_overrides(args))
    _resolve_training_paths(cfg)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.val_only:
        solver.val()
    else:
        solver.fit()


def build_parser():
    parser = argparse.ArgumentParser(description='Train or validate RegionBGC')
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--val-only', action='store_true', help='evaluate the validation set without training')
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, default=6, help='random seed (default: 6)')
    parser.add_argument('--train-data', '--train_data', dest='train_data', type=str)
    parser.add_argument('--train-features', '--train_features', dest='train_features', nargs='+')
    parser.add_argument('--val-data', '--val_data', dest='val_data', type=str)
    parser.add_argument('--val-features', '--val_features', dest='val_features', nargs='+')
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', type=str)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int)
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='single physical CUDA device id (default: 0)',
    )
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
