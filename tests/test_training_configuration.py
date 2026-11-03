import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.core import YAMLConfig
from src.solver.region_solver import RegionBGCSolver
from tools.train import build_parser, _build_config_overrides, _resolve_training_paths

ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigurationTests(unittest.TestCase):
    def configuration(self):
        return YAMLConfig(str(ROOT / 'configs/regionbgc/regionbgc.yml'))

    def solver(self):
        return RegionBGCSolver(SimpleNamespace(yaml_cfg={
            'bgc_eval': {'early_stopping': {'enable': True, 'patience': 10}},
        }))

    def test_defaults_align_main_without_evaluation_data(self):
        cfg = self.configuration()
        values = cfg.yaml_cfg
        self.assertEqual((values['seed'], values['epoches']), (6, 40))
        self.assertEqual(values['MultiScaleRegionFeatureEncoder']['dropout'], 0.2)
        self.assertEqual(values['RegionQueryDecoder']['dropout'], 0.2)
        self.assertEqual(cfg.criterion.det_pos_weight, 1.3)
        self.assertEqual(values['bgc_eval']['early_stopping'], {'enable': True, 'patience': 10})
        self.assertEqual(values['lr_scheduler']['milestones'], [16])
        self.assertEqual(values['lr_scheduler']['gamma'], 0.5)
        for loader in ('train_dataloader', 'val_dataloader'):
            self.assertEqual(values[loader]['batch_size'], 32)
            self.assertEqual(values[loader]['num_workers'], 0)
            self.assertFalse(values[loader]['persistent_workers'])
            self.assertIsNone(values[loader]['prefetch_factor'])
        self.assertNotIn('selection', values)
        self.assertNotIn('interval_ap_min_score', values['bgc_eval'])
        self.assertEqual(values['bgc_eval']['threshold'], 0.3)
        self.assertEqual(values['bgc_eval']['overlap_ratio'], 0.5)
        self.assertNotIn('test_dataloader', values)
        self.assertNotIn('type_test', values['bgc_eval'])
        self.assertNotIn('eval_test', values['bgc_eval'])
        args = build_parser().parse_args(['-c', str(ROOT / 'configs/regionbgc/regionbgc.yml')])
        self.assertEqual(_build_config_overrides(args)['seed'], 6)
        self.assertIsNone(args.resume)
        self.assertIsNone(args.tuning)

    def test_data_paths_are_resolved_and_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, features = root / 'train.json', root / 'train.h5'
            data.write_text('[]')
            features.touch()
            args = build_parser().parse_args([
                '-c', str(ROOT / 'configs/regionbgc/regionbgc.yml'),
                '--train-data', str(data), '--train-features', str(features),
                '--val-data', str(data), '--val-features', str(features),
            ])
            cfg = YAMLConfig(args.config, **_build_config_overrides(args))
            _resolve_training_paths(cfg)
            self.assertEqual(cfg.yaml_cfg['train_dataloader']['dataset']['data_file'], str(data))
            self.assertEqual(cfg.yaml_cfg['val_dataloader']['dataset']['features_files'], [str(features)])
            self.assertTrue(Path(cfg.output_dir).is_absolute())

    def test_ties_consume_patience_and_improvement_resets_it(self):
        solver = self.solver()
        with redirect_stdout(StringIO()):
            self.assertFalse(solver._update_validation_early_stopping(0.5, 0))
            for epoch in range(1, 10):
                self.assertFalse(solver._update_validation_early_stopping(0.5, epoch))
            self.assertTrue(solver._update_validation_early_stopping(0.5, 10))
            self.assertFalse(solver._update_validation_early_stopping(0.6, 11))
        self.assertEqual(solver._early_stopping_state['bad_epochs'], 0)
        self.assertEqual(solver._early_stopping_state['best_epoch'], 11)

    def test_early_stopping_uses_unrounded_values(self):
        solver = self.solver()
        with redirect_stdout(StringIO()):
            solver._update_validation_early_stopping(0.90001, 0)
            solver._update_validation_early_stopping(0.90002, 1)
        self.assertEqual(solver._early_stopping_state['best'], 0.90002)
        self.assertEqual(solver._early_stopping_state['bad_epochs'], 0)

    def test_resume_preserves_early_stopping_patience(self):
        solver = self.solver()
        solver.model = torch.nn.Linear(1, 1)
        solver.ema = solver.scaler = None
        with redirect_stdout(StringIO()):
            for epoch in range(10):
                solver._update_validation_early_stopping(0.5, epoch)
            state = solver.training_state_dict(9)
            restored = self.solver()
            restored.model = torch.nn.Linear(1, 1)
            restored.load_state_dict(state)
            self.assertTrue(restored._update_validation_early_stopping(0.5, 10))
        self.assertEqual(restored._early_stopping_state['bad_epochs'], 10)
        self.assertEqual(restored.last_epoch, 9)
        self.assertNotIn('best_selection', state)

    def test_fit_retains_every_epoch_including_the_early_stopping_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = self.solver()
            solver.cfg.epoches, solver.cfg.clip_max_norm, solver.cfg.log_step = 5, 0.1, 50
            solver.cfg.yaml_cfg['bgc_eval'].update(enable=True, eval_val=True,
                                                 early_stopping={'enable': True, 'patience': 2})
            solver.model = torch.nn.Linear(1, 1)
            solver.device, solver.last_epoch = torch.device('cpu'), -1
            solver.criterion = torch.nn.Identity()
            solver.ema = solver.scaler = None
            solver.output_dir = Path(directory)
            solver.lr_scheduler = SimpleNamespace(step=Mock(), state_dict=lambda: {})
            solver.train_dataloader = SimpleNamespace(sampler=Mock())
            solver.train = Mock()
            solver._bgc_eval_epoch = Mock(return_value={'interval_f1': 0.5})
            with patch('src.solver.region_solver.train_one_epoch', return_value={'loss': 1.0}), \
                 redirect_stdout(StringIO()):
                solver.fit()
            self.assertEqual(solver.lr_scheduler.step.call_count, 3)
            self.assertEqual(sorted(p.name for p in solver.output_dir.glob('epoch_*.pth')),
                             ['epoch_000.pth', 'epoch_001.pth', 'epoch_002.pth'])
            self.assertEqual(
                {p.name for p in solver.output_dir.glob('*.pth')},
                {'resume_latest.pth', 'epoch_000.pth', 'epoch_001.pth', 'epoch_002.pth'},
            )
            self.assertEqual(list(solver.output_dir.glob('best_*.pth')), [])
            state = torch.load(solver.output_dir / 'resume_latest.pth', weights_only=True)
            self.assertEqual(state['last_epoch'], 2)
            self.assertEqual(state['early_stopping']['bad_epochs'], 2)
            self.assertNotIn('best_selection', state)
            for epoch in range(3):
                artifact = torch.load(solver.output_dir / f'epoch_{epoch:03}.pth', weights_only=True)
                self.assertEqual(artifact['last_epoch'], epoch)
                self.assertTrue(artifact['format'].startswith('regionbgc-repvgg-deploy-'))
                self.assertNotIn('best_selection', artifact)
            self.assertEqual(list(solver.output_dir.glob('*.csv')), [])
            logs = [json.loads(line) for line in (solver.output_dir / 'log.txt').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in logs], [0, 1, 2])
            self.assertTrue(all(not any(key.startswith('val_') for key in row) for row in logs))

    def test_training_log_preserves_loss_and_learning_rate_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = self.solver()
            solver.output_dir = Path(directory)
            solver._write_training_log(0, {'loss': 1.23456789, 'lr': 0.000025})
            log = json.loads((solver.output_dir / 'log.txt').read_text())
            self.assertEqual(log, {'epoch': 0, 'train_loss': 1.23456789, 'train_lr': 0.000025})
            self.assertEqual(list(solver.output_dir.glob('*.csv')), [])

    def test_validation_reports_intervals_without_saving_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, features = root / 'val.json', root / 'val.h5'
            data.write_text(json.dumps([{'genome_id': 'g', 'orf_sequences': {'a': 'MA', 'b': 'MK'},
                                         'bgc_labels': [1, 1]}]))
            features.touch()
            solver = self.solver()
            solver.cfg.yaml_cfg.update(val_dataloader={'dataset': {
                'data_file': str(data), 'features_files': [str(features)], 'max_orfs': 2}})
            solver.cfg.yaml_cfg['bgc_eval'].update(enable=True, eval_val=True, interval_iou_thr=0.75,
                                                  dataloader={'num_workers': 0, 'batch_size': 1})
            solver.model, solver.device, solver.output_dir = torch.nn.Identity(), torch.device('cpu'), root
            solver.ema = None
            solver.cfg.postprocessor = lambda outputs, sizes: [{'segments': torch.tensor([[0, 2]]),
                                                                'scores': torch.tensor([0.9])}]
            manager = SimpleNamespace(embedding_dim=3, get_feature=lambda *a, **k: np.ones(3, dtype=np.float32))
            with patch('src.solver.region_solver.create_shared_features_manager', return_value=manager), \
                 patch.object(solver, '_eval_bgc_type_metrics', return_value={'f1': 0.8}), \
                 redirect_stdout(StringIO()) as console:
                metrics = solver._bgc_eval_epoch(0)
            self.assertEqual(metrics['interval_f1'], 1.0)
            self.assertEqual(metrics['type_f1'], 0.8)
            self.assertIn('Intervals: P=1.000 | R=1.000 | F1=1.000', console.getvalue())
            self.assertIn('Product types: f1=0.800', console.getvalue())
            self.assertEqual(list(root.glob('*.csv')), [])
            self.assertNotIn('ap50', metrics)
            self.assertNotIn('ap75', metrics)
            self.assertNotIn('precision', metrics)
            self.assertNotIn('auroc', metrics)
            self.assertEqual(list(root.glob('*.pth')), [])


if __name__ == '__main__':
    unittest.main()
