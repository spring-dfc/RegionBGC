# RegionBGC adaptation for BGC region prediction; upstream attribution is retained.
"""by lyuwenyu
"""

import torch
import tempfile
import yaml

from pathlib import Path
from typing import Dict

from src.core import BaseConfig
from src.misc import distributed
from src.models.deploy import DEPLOY_FORMATS, build_deploy_checkpoint
from src.models.interval_ops import validate_interval_checkpoint


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg
        self.optimizer = None
        self.lr_scheduler = None

    def setup(self, ):
        '''Avoid instantiating unnecessary classes
        '''
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        self.model = distributed.wrap_model(
            cfg.model.to(device),
            find_unused_parameters=bool(cfg.yaml_cfg.get('find_unused_parameters', True)),
            sync_bn=bool(cfg.yaml_cfg.get('sync_bn', True)),
        )
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / 'resolved_config.yml').open('w') as handle:
            resolved = {key: value for key, value in cfg.yaml_cfg.items() if key != '__include__'}
            yaml.safe_dump(resolved, handle, sort_keys=False)


    def train(self, ):
        self.setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        seed = int(self.cfg.yaml_cfg.get('seed', 0))
        self.train_dataloader = distributed.wrap_loader(
            self.cfg.train_dataloader,
            shuffle=self.cfg.train_dataloader.shuffle,
            seed=seed,
        )
        self.val_dataloader = distributed.wrap_loader(
            self.cfg.val_dataloader,
            shuffle=self.cfg.val_dataloader.shuffle,
            seed=seed,
        )


    def eval(self, ):
        self.setup()
        self.val_dataloader = self.cfg.val_dataloader

        if self.cfg.resume:
            print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)


    def inference_state_dict(self, last_epoch=None):
        """Return fused EMA interval weights with their resolved model configuration."""
        module = self.ema.module if self.ema is not None else distributed.de_parallel(self.model)
        source = 'ema' if self.ema is not None else 'model'
        epoch = self.last_epoch if last_epoch is None else last_epoch
        artifact = build_deploy_checkpoint(module, self.cfg.yaml_cfg, epoch, source)
        return artifact


    def training_state_dict(self, last_epoch):
        """Return the complete state required for exact training resumption."""
        state = {
            'model': distributed.de_parallel(self.model).state_dict(),
            'last_epoch': int(last_epoch),
            'feature_layout': 'BCL',
        }
        representation = getattr(distributed.de_parallel(self.model), 'interval_representation', None)
        if representation is not None:
            state['interval_representation'] = representation
        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        if hasattr(self, '_early_stopping_state'):
            state['early_stopping'] = dict(self._early_stopping_state)
        return state


    def load_state_dict(self, state):
        '''load state dict
        '''
        if state.get('format') in DEPLOY_FORMATS:
            raise ValueError('Use resume_latest.pth to resume training; fused checkpoints are for prediction.')
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            module = distributed.de_parallel(self.model)
            validate_interval_checkpoint(state, state['model'])
            module.load_state_dict(state['model'], strict=True)
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')
        if isinstance(state.get('early_stopping'), dict):
            self._early_stopping_state = dict(state['early_stopping'])


    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(state)

    def load_tuning_state(self, path,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu', weights_only=True)

        module = distributed.de_parallel(self.model)

        candidate = state.get('ema', {}).get('module', state.get('model', state))
        validate_interval_checkpoint(state, candidate)
        # Match only compatible interval-model layers.
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


    def fit(self, ):
        raise NotImplementedError('')

    def val(self, ):
        raise NotImplementedError('')


    def save(self, path, last_epoch=None):
        """Save the latest training state and one fused checkpoint per epoch."""
        if not distributed.is_main_process():
            return
        path = Path(path)
        epoch = self.last_epoch if last_epoch is None else last_epoch
        state = self.training_state_dict(epoch)
        artifact = self.inference_state_dict(last_epoch=epoch)
        artifact['metadata']['source_checkpoint'] = path.name
        deploy_path = path.with_name(f'epoch_{epoch:03}.pth')
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = []
        try:
            # Finish both serializations before replacing the previous files.
            for payload, destination in ((state, path), (artifact, deploy_path)):
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + destination.name,
                                                 suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    pending.append((temporary, destination))
                    distributed.save_on_master(payload, handle)
            for temporary, destination in pending:
                temporary.replace(destination)
        finally:
            for temporary, _ in pending:
                temporary.unlink(missing_ok=True)
        print(f'[Save] Training: {path} | Fused deployment: {deploy_path}')
