"""Current checkpoint, optimizer and configuration round-trip checks."""

from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver.solver import BaseSolver
from src.models.deploy import build_deploy_checkpoint
from src.models.loading import load_evaluation_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/regionbgc/regionbgc.yml'


class CheckpointRoundtripTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(6)

    def test_resolved_configuration_roundtrip(self):
        expected = dict(YAMLConfig(str(CONFIG)).yaml_cfg)
        expected.pop('__include__', None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.yml'
            path.write_text(yaml.safe_dump(expected))
            restored = YAMLConfig(str(path))
            self.assertEqual(restored.yaml_cfg, expected)
            self.assertEqual(type(restored.model).__name__, 'RegionBGC')
            self.assertEqual(restored.criterion.det_pos_weight, 1.3)

    def test_configuration_loads_do_not_reuse_previous_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'first.yml', Path(directory) / 'second.yml'
            first.write_text('model_setting: 1\n')
            second.write_text('independent_setting: 2\n')
            a, b = load_config(str(first)), load_config(str(second))
        self.assertEqual(a, {'model_setting': 1})
        self.assertEqual(b, {'independent_setting': 2})
        self.assertIsNot(a, b)

    def test_fused_deployment_restores_identical_tensors_and_predictions(self):
        cfg = YAMLConfig(str(CONFIG))
        model = cfg.model.eval()
        artifact = build_deploy_checkpoint(model, cfg.yaml_cfg, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(artifact, path)
            restored = YAMLConfig(str(CONFIG)).model
            load_evaluation_checkpoint(restored, path)
            for name, value in artifact['model'].items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))
            features = torch.randn(2, 960, 128)
            with torch.no_grad():
                expected, actual = model(features), restored(features)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=1e-4, rtol=1e-4)

    def test_training_state_restores_parameters_ema_and_optimizer(self):
        cfg = YAMLConfig(str(CONFIG)); solver = BaseSolver(cfg)
        solver.model, solver.optimizer, solver.ema = cfg.model, cfg.optimizer, cfg.ema
        solver.lr_scheduler = solver.scaler = None
        solver.last_epoch = 3
        sum(parameter.sum() * 1e-6 for parameter in solver.model.parameters()).backward()
        solver.optimizer.step(); solver.ema.update(solver.model)
        state = solver.training_state_dict(3)
        fresh = YAMLConfig(str(CONFIG)); restored = BaseSolver(fresh)
        restored.model, restored.optimizer, restored.ema = fresh.model, fresh.optimizer, fresh.ema
        restored.lr_scheduler = restored.scaler = None
        restored.load_state_dict(state)
        self.assertEqual(restored.last_epoch, 3)
        self.assertEqual(restored.ema.updates, solver.ema.updates)
        for name, value in solver.model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.model.state_dict()[name]))
        for name, value in solver.ema.module.state_dict().items():
            self.assertTrue(torch.equal(value, restored.ema.module.state_dict()[name]))
        a, b = solver.optimizer.state_dict(), restored.optimizer.state_dict()
        self.assertEqual(a['param_groups'], b['param_groups'])
        for index, fields in a['state'].items():
            for name, value in fields.items():
                self.assertTrue(torch.equal(value, b['state'][index][name]))

    def test_incompatible_convolution_shape_raises(self):
        model = torch.nn.Conv1d(3, 4, 1)
        state = dict(model.state_dict()); state['weight'] = state['weight'].unsqueeze(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'; torch.save({'model': state}, path)
            with self.assertRaises(RuntimeError):
                load_evaluation_checkpoint(model, path)


if __name__ == '__main__':
    unittest.main()
