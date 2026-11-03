"""Native interval model, prediction boundaries and checkpoint regression checks."""

from pathlib import Path
import tempfile
import json
import numpy as np
import unittest

import torch
from torch import nn

from src.core import YAMLConfig
from src.solver.solver import BaseSolver
from src.models.encoder import RepVGGBlock
from src.models.interval_ops import validate_interval_checkpoint
from src.models.loading import load_evaluation_checkpoint
from src.models.postprocessor import RegionBGCPostProcessor
from utils.bgc_inference import predict_valid_sequences

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/regionbgc/regionbgc.yml'


class NativeModelAndDeployTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(6)

    def test_main_model_uses_native_sequences_and_selected_auxiliary_queries(self):
        cfg = YAMLConfig(str(CONFIG))
        model = cfg.model.train()
        orf_features = torch.randn(2, 960, 128)
        features = model.backbone(orf_features)
        self.assertEqual([tuple(f.shape) for f in features],
                         [(2, 960, 128), (2, 960, 64), (2, 960, 32)])
        outputs = model(orf_features)
        self.assertEqual(outputs['pred_intervals'].shape, (2, 64, 2))
        self.assertEqual(outputs['pred_bgc_types'].shape, (2, 64, 6))
        self.assertEqual(outputs['enc_aux_outputs'][0]['pred_intervals'].shape, (2, 64, 2))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9170713)
        self.assertFalse(any(isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.AvgPool2d))
                             for m in model.modules()))
        self.assertNotIn('test_dataloader', cfg.yaml_cfg)
        self.assertNotIn('type_test', cfg.yaml_cfg['bgc_eval'])

    def test_mixed_prediction_lengths_crop_padding_before_coordinate_recovery(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, orf_features):
                self.calls.append(tuple(orf_features.shape))
                count = orf_features.shape[0]
                return {'pred_logits': torch.zeros(count, 1, 1),
                        'pred_intervals': torch.tensor([.1875, .125]).repeat(count, 1, 1)}

        model = Recorder()
        features = torch.randn(3, 128, 4)
        mask = torch.arange(128)[None] < torch.tensor([64, 128, 64])[:, None]
        results, lengths = predict_valid_sequences(model, RegionBGCPostProcessor(), features, mask)
        self.assertEqual(model.calls, [(2, 4, 64), (1, 4, 128)])
        self.assertEqual(lengths.tolist(), [64, 128, 64])
        self.assertEqual([r['segments'].tolist() for r in results],
                         [[[8, 16]], [[16, 32]], [[8, 16]]])

    def test_every_save_fuses_ema_and_preserves_training_parameters_and_rng(self):
        cfg = YAMLConfig(str(CONFIG))
        model, criterion, optimizer, ema = cfg.model, cfg.criterion, cfg.optimizer, cfg.ema
        model.train()
        orf_features = torch.randn(2, 960, 128)
        targets = [dict(labels=torch.tensor([0]), intervals=torch.tensor([[.5, .25]]),
                        bgc_type_vec=torch.tensor([[1., 0., 0., 0., 1., 0.]])),
                   dict(labels=torch.empty(0, dtype=torch.long), intervals=torch.empty(0, 2),
                        bgc_type_vec=torch.empty(0, 6))]
        losses = criterion(model(orf_features), targets)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        optimizer.step()
        ema.update(model)
        solver = BaseSolver(cfg)
        solver.model, solver.optimizer, solver.ema = model, optimizer, ema
        solver.scaler = solver.lr_scheduler = None
        solver.last_epoch = 0
        before_rng = torch.get_rng_state()
        parameter_ids = [id(p) for p in model.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume_latest.pth'
            solver.save(path)
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertEqual(parameter_ids, [id(p) for p in model.parameters()])
            self.assertTrue(model.training)
            self.assertTrue(all(hasattr(m, 'conv1') for m in model.modules()
                                if isinstance(m, RepVGGBlock)))
            self.assertFalse(path.with_name('resume_latest_deploy.pth').exists())
            checkpoint = path.with_name('epoch_000.pth')
            artifact = torch.load(checkpoint, weights_only=True)
            self.assertEqual(artifact['feature_layout'], 'BCL')
            self.assertEqual(artifact['metadata']['weights_source'], 'ema')
            self.assertEqual(artifact['metadata']['fused_blocks'], 12)
            restored = YAMLConfig(str(CONFIG)).model
            load_evaluation_checkpoint(restored, checkpoint)
            restored.eval()
            with torch.no_grad():
                expected, actual = ema.module(orf_features), restored(orf_features)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=1e-4, rtol=1e-4)
            resumed = BaseSolver(cfg)
            resumed.model = YAMLConfig(str(CONFIG)).model
            resumed.ema = resumed.scaler = resumed.optimizer = resumed.lr_scheduler = None
            resumed.resume(path)
            self.assertEqual(resumed.last_epoch, 0)

    def test_prepared_windows_keep_sample_ids_feature_ids_and_offsets_separate(self):
        from src.predict.dataset import PredictionDataset
        class Features:
            embedding_dim = 3
            def get_feature(self, oid, gid, copy=False):
                if gid != 'parent' or oid not in ['orf_a', 'orf_b']:
                    return None
                return np.ones(3, dtype=np.float32)
        samples = [dict(genome_id='window_a', original_genome_id='parent',
                        orf_sequences={'orf_a': 'MA', 'orf_b': 'MK'},
                        bgc_labels=[0, 1], segment_start=20, segment_end=22,
                        total_orfs_in_genome=200)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_text(json.dumps(samples))
            item = PredictionDataset(str(path), Features(), use_sliding_window=False)[0]
        self.assertEqual(item['genome_id'], 'window_a')
        self.assertEqual(item['original_genome_id'], 'parent')
        self.assertEqual(item['segment_start'], 20)
        self.assertEqual(item['segment_end'], 22)
        self.assertEqual(item['total_orfs_in_genome'], 200)
        self.assertEqual(item['padding_mask'].sum().item(), 2)

    def test_four_coordinate_checkpoints_are_rejected_before_loading(self):
        state = {'decoder.enc_interval_head.layers.2.weight': torch.zeros(4, 256)}
        with self.assertRaises(ValueError):
            validate_interval_checkpoint({'model': state}, state)


class HungarianTypeEvaluationTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(criterion=YAMLConfig(str(CONFIG)).criterion)
        torch.set_num_threads(1)

    def evaluate(self, outputs, targets, batch_size=32):
        from unittest.mock import patch
        from src.evaluation.test_evaluator import evaluate_product_types

        class FixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = 0

            def forward(self, samples):
                start, end = self.offset, self.offset + len(samples)
                self.offset = end
                return {key: value[start:end] for key, value in outputs.items()}

        dataset = [(torch.ones(3, 8), target) for target in targets]
        with patch('src.evaluation.test_evaluator._load_model',
                   return_value=(FixedModel(), self.cfg, 'fixture')), \
             patch('src.evaluation.test_evaluator._InMemoryRegionBGCDataset', return_value=dataset):
            return evaluate_product_types(
                config=str(CONFIG), checkpoint='unused.pth', data_file='unused.json',
                features_manager=None, device=torch.device('cpu'),
                batch_size=batch_size, num_workers=0,
            )

    def test_global_assignment_and_low_iou_pairs_use_unmodified_class_scores(self):
        from src.models.matcher import match_queries_by_hungarian
        from src.solver.region_solver import RegionBGCSolver
        from types import SimpleNamespace
        outputs = {
            'pred_logits': torch.full((1, 2, 1), -8.),
            'pred_intervals': torch.tensor([[[.45, .2], [.25, .2]]]),
            'pred_bgc_types': torch.tensor([[[9., -9., -9., -9., -9., -9.],
                                             [-9., 9., -9., -9., -9., -9.]]]),
        }
        target = {'labels': torch.zeros(2, dtype=torch.long),
                  'intervals': torch.tensor([[.4, .2], [.6, .2]]),
                  'bgc_type_vec': torch.tensor([[0., 1., 0., 0., 0., 0.],
                                                 [1., 0., 0., 0., 0., 0.]])}
        matches = match_queries_by_hungarian(self.cfg.criterion.matcher, outputs, [target])
        self.assertEqual(matches, [[1, 0]])
        metrics, counts, _ = self.evaluate(outputs, [target])
        self.assertEqual(metrics, dict(precision=1., recall=1., f1=1., auroc=1.))
        self.assertEqual(counts['matched'], 2)
        self.assertEqual(counts['coverage'], 1.)
        solver = RegionBGCSolver(SimpleNamespace(yaml_cfg={}))
        solver.device, solver.criterion = torch.device('cpu'), self.cfg.criterion
        solver.val_dataloader = [(torch.ones(1, 3, 8), [target])]
        class Model(nn.Module):
            def forward(self, samples):
                return outputs
        validation = solver._eval_bgc_type_metrics(Model())
        self.assertEqual({key: validation[key] for key in metrics}, metrics)
        self.assertEqual(validation['matched'], 2)

    def test_unmatched_ground_truth_only_affects_coverage(self):
        outputs = {'pred_logits': torch.full((1, 1, 1), -8.),
                   'pred_intervals': torch.tensor([[[.2, .1]]]),
                   'pred_bgc_types': torch.tensor([[[9., -9., -9., -9., -9., -9.]]])}
        target = {'labels': torch.zeros(3, dtype=torch.long),
                  'intervals': torch.tensor([[.2, .1], [.5, .1], [.8, .1]]),
                  'bgc_type_vec': torch.tensor([[1., 0., 0., 0., 0., 0.],
                                                 [0., 1., 0., 0., 0., 0.],
                                                 [0., 1., 0., 0., 0., 0.]])}
        metrics, counts, _ = self.evaluate(outputs, [target])
        self.assertEqual(metrics['f1'], 1.)
        self.assertEqual((counts['regions'], counts['matched'], counts['unmatched']), (3, 1, 2))
        self.assertAlmostEqual(counts['coverage'], 1 / 3)

    def test_batch32_processes_background_batch_and_last_partial_batch(self):
        empty = {'labels': torch.empty(0, dtype=torch.long),
                 'intervals': torch.empty(0, 2), 'bgc_type_vec': torch.empty(0, 6)}
        positive = {'labels': torch.zeros(1, dtype=torch.long),
                    'intervals': torch.tensor([[.8, .1]]),
                    'bgc_type_vec': torch.tensor([[1., 0., 0., 0., 0., 0.]])}
        outputs = {'pred_logits': torch.full((33, 1, 1), -8.),
                   'pred_intervals': torch.tensor([[[.1, .1]]]).repeat(33, 1, 1),
                   'pred_bgc_types': torch.tensor([[[9., -9., -9., -9., -9., -9.]]]).repeat(33, 1, 1)}
        for batch_size in [1, 32]:
            with self.subTest(batch_size=batch_size):
                metrics, counts, _ = self.evaluate(outputs, [empty] * 32 + [positive], batch_size)
                self.assertEqual(metrics['f1'], 1.)
                self.assertEqual((counts['samples'], counts['regions'], counts['matched']), (33, 1, 1))

    def test_online_dataset_supplies_detection_labels_for_matching(self):
        from types import SimpleNamespace
        from src.evaluation.test_evaluator import _InMemoryRegionBGCDataset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.json'
            path.write_text(json.dumps([
                {'genome_id': 'g', 'orf_sequences': {'a': 'MA', 'b': 'MK'},
                 'bgc_labels': [1, 1], 'bgc_class': ['PKS']},
                {'genome_id': 'g', 'orf_sequences': {'a': 'MA', 'b': 'MK'},
                 'bgc_labels': [0, 0], 'bgc_class': []},
            ]))
            features = SimpleNamespace(embedding_dim=3, get_feature=lambda *a, **k: np.ones(3))
            dataset = _InMemoryRegionBGCDataset(str(path), features)
            self.assertEqual(dataset[0][1]['labels'].tolist(), [0])
            self.assertEqual(dataset[1][1]['labels'].tolist(), [])


if __name__ == '__main__':
    unittest.main()
