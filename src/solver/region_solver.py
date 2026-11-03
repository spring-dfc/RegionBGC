# RegionBGC adaptation for BGC region prediction; upstream attribution is retained.
"""RegionBGC training with per-epoch weights and interval validation.

Adapted from the original training solver by lyuwenyu.
"""

import datetime
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.predict.dataset import PredictionDataset, collate_prediction_batch
from src.metrics.regions import (
    PredInterval, build_gt_intervals_from_labels,
    compute_interval_prf_at_score, normalize_pred_segments,
)
from src.metrics.classification import compute_multilabel_micro_metrics
from src.utils.shared_features import create_shared_features_manager
from src.utils.bgc_inference import predict_valid_sequences
from src.models.matcher import match_queries_by_hungarian

from .train_engine import train_one_epoch
from .solver import BaseSolver


class RegionBGCSolver(BaseSolver):
    def fit(self):
        print('Start training')
        self.train()
        args = self.cfg
        eval_cfg = args.yaml_cfg.get('bgc_eval', {})
        early = eval_cfg.get('early_stopping', {})
        if early.get('enable', False):
            if int(early.get('patience', 10)) < 1:
                raise ValueError('early_stopping.patience must be at least 1')
            if not eval_cfg.get('enable', True) or not eval_cfg.get('eval_val', True):
                raise ValueError('Early stopping requires validation evaluation every epoch')

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer,
                self.device, epoch, args.clip_max_norm, print_freq=args.log_step,
                ema=self.ema, scaler=self.scaler,
            )
            self.lr_scheduler.step()
            val_metrics = {}
            if eval_cfg.get('enable', False):
                val_metrics = self._bgc_eval_epoch(epoch) or {}
            should_stop = self._update_validation_early_stopping(
                val_metrics.get('interval_f1'), epoch,
            )
            if self.output_dir:
                self.save(self.output_dir / 'resume_latest.pth', last_epoch=epoch)
                self._write_training_log(epoch, train_stats)
            if should_stop:
                break
        elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print(f'Training time {elapsed}')

    def _write_training_log(self, epoch, train_stats):
        """Keep full-precision training losses and LR; validation is console-only."""
        log = {'epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()}}
        with (self.output_dir / 'log.txt').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(log) + '\n')

    def _update_validation_early_stopping(self, value, epoch):
        """Track validation interval F1 for patience without selecting weights."""
        cfg = self.cfg.yaml_cfg.get('bgc_eval', {}).get('early_stopping', {})
        if not cfg.get('enable', False):
            return False
        if value is None or not np.isfinite(value):
            print('[EarlyStop] Validation interval F1 unavailable; patience unchanged.')
            return False
        patience = int(cfg.get('patience', 10))
        state = getattr(self, '_early_stopping_state', None)
        if (not isinstance(state, dict) or state.get('monitor') != 'val_interval_f1'
                or state.get('mode') != 'max'):
            state = {'monitor': 'val_interval_f1', 'mode': 'max', 'best': None,
                     'best_epoch': -1, 'bad_epochs': 0, 'last_epoch': -1}
        if epoch <= state['last_epoch']:
            return state['bad_epochs'] >= patience
        value = float(value)
        if state['best'] is None or value > state['best']:
            state.update(best=value, best_epoch=int(epoch), bad_epochs=0)
        else:
            state['bad_epochs'] += 1
        state['last_epoch'] = int(epoch)
        self._early_stopping_state = state
        print(f"[EarlyStop] Epoch {epoch}: interval F1={value:.3f}, "
              f"highest={state['best']:.3f}, "
              f"patience={state['bad_epochs']}/{patience}")
        return state['bad_epochs'] >= patience

    @torch.no_grad()
    def _eval_bgc_type_metrics(self, module) -> Dict[str, float]:
        """Validation classification on Hungarian-matched pairs."""
        if getattr(self, 'val_dataloader', None) is None:
            return {}
        matcher = self.criterion.matcher
        all_logits, all_targets = [], []
        total_gt = 0
        for samples, targets in self.val_dataloader:
            samples = samples.to(self.device)
            targets = [{key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                        for key, value in target.items()} for target in targets]
            outputs = module(samples)
            if 'pred_bgc_types' not in outputs:
                return {}
            assignments = match_queries_by_hungarian(matcher, outputs, targets)
            for batch_index, matches in enumerate(assignments):
                total_gt += len(matches)
                for gt_index, query_index in enumerate(matches):
                    if query_index is None:
                        continue
                    all_logits.append(outputs['pred_bgc_types'][batch_index, query_index].cpu())
                    all_targets.append(targets[batch_index]['bgc_type_vec'][gt_index].cpu())
        if not all_logits:
            return {}
        metrics = compute_multilabel_micro_metrics(
            torch.stack(all_logits), torch.stack(all_targets), threshold=0.5,
        )
        matched = len(all_logits)
        metrics.update(matched=matched, total_gt=total_gt, unmatched=total_gt - matched,
                       coverage=matched / total_gt)
        return metrics

    @torch.no_grad()
    def _bgc_eval_epoch(self, epoch: int):
        """Evaluate validation intervals and product types at fixed thresholds."""
        cfg = self.cfg.yaml_cfg
        eval_cfg = cfg.get('bgc_eval', {})
        if not eval_cfg.get('enable', True) or not eval_cfg.get('eval_val', True):
            return {}
        ds_cfg = cfg.get('val_dataloader', {}).get('dataset', {})
        data_file = ds_cfg.get('data_file')
        features_files = ds_cfg.get('features_files', [])
        features_file = features_files[0] if features_files else None
        for name, value in (('data_file', data_file), ('features_file', features_file)):
            if not value or not Path(value).is_file():
                raise FileNotFoundError(f'Missing validation {name}: {value}')

        cache = getattr(self, '_shared_features_cache', None)
        if cache is None:
            cache = self._shared_features_cache = {}
        if features_file not in cache:
            cache[features_file] = create_shared_features_manager(features_file)
        dataset = PredictionDataset(
            data_file=data_file, features_manager=cache[features_file],
            max_orfs=int(ds_cfg.get('max_orfs', 128)), use_sliding_window=False,
        )
        if len(dataset) == 0:
            raise ValueError(f'Validation dataset is empty: {data_file}')
        loader_cfg = {'batch_size': 16, 'num_workers': 4, 'pin_memory': True,
                      'persistent_workers': True, 'prefetch_factor': 2}
        loader_cfg.update(eval_cfg.get('dataloader', {}))
        loader_cfg.update(eval_cfg.get('val_dataloader', {}))
        batch_size = int(loader_cfg['batch_size'])
        workers = int(loader_cfg['num_workers'])
        if batch_size <= 0 or workers < 0:
            raise ValueError('Evaluation batch_size must be positive and num_workers non-negative')
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=bool(loader_cfg['pin_memory']),
            persistent_workers=bool(loader_cfg['persistent_workers']) if workers else False,
            prefetch_factor=(loader_cfg['prefetch_factor'] or 2) if workers else None,
            collate_fn=collate_prediction_batch,
        )
        module = self.ema.module if self.ema else self.model
        module.eval()
        predictions: List[PredInterval] = []
        ground_truth = {}
        sample_counter = 0
        topk = int(eval_cfg.get('interval_topk', 64))
        for batch in loader:
            results, lengths = predict_valid_sequences(
                module, self.cfg.postprocessor, batch['features'].to(self.device),
                batch['padding_mask'],
            )
            for index, result in enumerate(results):
                length = int(lengths[index].item())
                sample_id = f"validation:{batch['genome_id'][index]}:{sample_counter}"
                sample_counter += 1
                labels = batch['bgc_labels'][index][:length].cpu().numpy().astype(float)
                ground_truth[sample_id] = build_gt_intervals_from_labels(labels)
                segments, scores = result.get('segments'), result.get('scores')
                if segments is None or scores is None:
                    continue
                candidates = normalize_pred_segments(
                    segments.detach().cpu().numpy(), scores.detach().cpu().numpy(), W=length,
                )
                candidates.sort(key=lambda item: float(item[1]), reverse=True)
                if topk > 0:
                    candidates = candidates[:topk]
                predictions.extend(PredInterval(sample_id, float(score), segment)
                                   for segment, score in candidates)
        score_thr = float(eval_cfg.get('interval_score_thr', 0.3))
        iou_thr = float(eval_cfg.get('interval_iou_thr', 0.75))
        metrics = compute_interval_prf_at_score(
            predictions, ground_truth, iou_thr=iou_thr, score_thr=score_thr,
        )
        metrics['interval_score_thr'] = score_thr
        types = self._eval_bgc_type_metrics(module) or {}
        metrics.update({f'type_{key}': value for key, value in types.items()
                        if key in ('precision', 'recall', 'f1', 'auroc')})
        print(f'\n[RegionBGC] Epoch {epoch} validation '
              f'(score={score_thr:.3f}, IoU={iou_thr:.3f})')
        print(f"Intervals: P={metrics['interval_precision']:.3f} | "
              f"R={metrics['interval_recall']:.3f} | F1={metrics['interval_f1']:.3f}")
        if types:
            print('Product types: ' + ' | '.join(
                f'{key}={types[key]:.3f}' for key in ('precision', 'recall', 'f1', 'auroc')
                if key in types
            ))
        return metrics

    def val(self):
        self.eval()
        return self._bgc_eval_epoch(epoch=getattr(self, 'last_epoch', 0))
