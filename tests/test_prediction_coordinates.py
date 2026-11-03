import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.predict.dataset import PredictionDataset, collate_prediction_batch
from src.predict.feature_extractor import _iter_unique_orfs
from src.predict.memory_features import InMemoryFeaturesManager
from src.predict.prepare_dataset import build_samples_from_fasta
from src.data.bgc.bgc_dataset import RegionBGCDataset
from src.evaluation.test_evaluator import _InMemoryRegionBGCDataset
from tools.extract_features.extract_features import iter_unique_orfs
from utils.bgc_data import sample_orf_ids
from src.predict.predictor import (
    _resolve_eval_thresholds,
    build_orf_position_map,
    evaluate_model,
    run_prediction,
)


class ParentFeatures:
    embedding_dim = 3

    def get_feature(self, orf_id, genome_id=None, copy=False):
        if genome_id != "parent":
            raise AssertionError("Features must use the original genome ID")
        return np.full(3, int(orf_id.split("_")[-1]), dtype=np.float32)


class PredictionCoordinateTests(unittest.TestCase):
    def test_real_sequence_audit_accepts_optional_merged_json(self):
        from src.predict import run
        from src.predict.audit_prediction_sequences import audit
        from src.predict.build_prediction_json import build_prediction_json
        from src.predict.genbank_writer import write_prediction_genbanks
        from src.predict.prepare_dataset import convert_fasta_file_to_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / 'sample'; sample.mkdir()
            dna = root / 'prepared_inputs/nucleotide/sample.fna'
            faa = root / 'prepared_inputs/protein/sample.faa'
            dna.parent.mkdir(parents=True); faa.parent.mkdir(parents=True)
            original = root / 'sample.fa'
            original.write_text('>original\nGTGAAATAACCCTTAGGGCAT\n')
            dna.write_text('>sample_1\nGTGAAATAACCCTTAGGGCAT\n')
            faa.write_text('>sample_1_1 # 1 # 9 # 1 # ID=1_1;partial=00;start_type=GTG;\nMK*\n'
                           '>sample_1_2 # 13 # 21 # -1 # ID=1_2;partial=00;start_type=ATG;\nMP*\n')
            faa.with_suffix('.genes').write_text(
                'DEFINITION seqnum=1;seqlen=21;seqhdr="sample_1";transl_table=11\n'
                'FEATURES             Location/Qualifiers\n'
                '     CDS             1..9\n                     /note="ID=1_1;partial=00;"\n'
                '     CDS             complement(13..21)\n                     /note="ID=1_2;partial=00;"\n//\n')
            convert_fasta_file_to_dataset(str(faa), str(sample / 'prediction_dataset.json'))
            (sample / 'orf_predictions.tsv').write_text(
                'ORF_ID\tProb\tPred\nsample_1_1\t0.9\t1\nsample_1_2\t0.9\t1\n')
            csv_path = sample / 'bgc_regions.csv'
            csv_path.write_text('genome,class_name,confidence,start_cds,end_cds,cds_ids,bgc_range,cds_len,nt_len\n'
                                'sample_1,RiPP,0.9,1,2,sample_1_1 sample_1_2,sample_1:1-21,2,21\n')
            build_prediction_json(str(csv_path), str(dna), str(sample / 'bgc_regions.json'), 'sample')
            write_prediction_genbanks(dna_fasta=str(dna), protein_faa=str(faa),
                                      regions_csv=str(csv_path), output_dir=str(sample), sample_name='sample')
            original_csv = csv_path.read_bytes()
            original_gbk = {p: p.read_bytes() for p in sample.rglob('*.gbk')}
            for save_json in (False, True):
                run.merge_sample_prediction_outputs(root, ['sample'], save_json=save_json)
                result = audit(original, root, check_merged_json=save_json)
                self.assertEqual(result['status'], 'PASS', result['errors'])
                self.assertEqual(result['coverage_results']['merged_prediction_tsv']['count'], 2)
                self.assertEqual(csv_path.read_bytes(), original_csv)
                self.assertEqual({p: p.read_bytes() for p in original_gbk}, original_gbk)
            (root / 'merged_bgc_regions.json').unlink()
            self.assertEqual(audit(original, root, check_merged_json=True)['status'], 'FAIL')

    def _dataset(self, directory, count, max_orfs, base_start=40):
        sample = {
            "genome_id": "prepared_window",
            "original_genome_id": "parent",
            "orf_sequences": {f"orf_{i}": "MA" for i in range(count)},
            "bgc_labels": [0] * count,
            "segment_start": base_start,
            "segment_end": base_start + count,
            "total_orfs_in_genome": 100,
        }
        path = Path(directory) / "windows.json"
        path.write_text(json.dumps([sample]), encoding="utf-8")
        return PredictionDataset(str(path), ParentFeatures(), max_orfs=max_orfs)

    def test_short_prepared_window_preserves_metadata_and_features(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(directory, 16, 128)
            item = dataset[0]
        self.assertEqual(item["genome_id"], "prepared_window")
        self.assertEqual(item["original_genome_id"], "parent")
        self.assertEqual((item["segment_start"], item["segment_end"]), (40, 56))
        self.assertEqual(item["total_orfs_in_genome"], 100)
        self.assertEqual(item["padding_mask"].sum().item(), 16)
        torch.testing.assert_close(item["features"][:16, 0], torch.arange(16).float())

    def test_subwindows_preserve_parent_offsets_ids_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(directory, 19, 8)
            items = [dataset[i] for i in range(len(dataset))]
        self.assertEqual(
            [(x["segment_start"], x["segment_end"]) for x in items],
            [(40, 48), (44, 52), (48, 56), (51, 59)],
        )
        covered = set()
        for item in items:
            start, end = item["segment_start"], item["segment_end"]
            covered.update(range(start, end))
            self.assertEqual(item["genome_id"], f"prepared_window_seg_{start}_{end}")
            self.assertEqual(item["original_genome_id"], "parent")
            self.assertEqual(item["total_orfs_in_genome"], 100)
            self.assertEqual(item["orf_ids"][0], f"orf_{start - 40}")
            self.assertEqual(item["features"][0, 0].item(), start - 40)
        self.assertEqual(covered, set(range(40, 59)))

    def test_subwindow_predictions_restore_global_coordinates(self):
        class Model(torch.nn.Module):
            def forward(self, features):
                self.features = features
                return {}

        def postprocessor(outputs, lengths):
            self.assertEqual(lengths.tolist(), [8, 8])
            return [
                {"segments": torch.tensor([[2.0, 6.0]]), "scores": torch.tensor([0.9])}
                for _ in lengths
            ]

        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(directory, 10, 8)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_prediction_batch)
            model = Model()
            fused = evaluate_model(model, postprocessor, loader, torch.device("cpu"), 0.3, 0.5)
        self.assertEqual(tuple(model.features.shape), (2, 3, 8))
        self.assertEqual(set(fused), {"parent"})
        result = fused["parent"]
        self.assertEqual(len(result["predictions"]), 100)
        self.assertEqual(
            [(x["start"], x["end"]) for x in result["intervals"]],
            [(42, 46), (44, 48)],
        )
        self.assertEqual(
            [(x["start"], x["end"]) for x in result["final_regions"]], [(42, 48)]
        )
        self.assertEqual(result["orf_ids"][40:50], [f"orf_{i}" for i in range(10)])

    def test_position_maps_merge_overlapping_windows(self):
        samples = [
            {"genome_id": "window_a", "original_genome_id": "parent",
             "orf_ids": ["a", "b"], "orf_positions": [[1, 9], [11, 19]]},
            {"genome_id": "window_b", "original_genome_id": "parent",
             "orf_ids": ["b", "c"], "orf_positions": [[11, 19], [21, 29]]},
            {"genome_id": "other", "orf_ids": ["a"], "orf_positions": [[101, 109]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positions.json"
            path.write_text(json.dumps(samples), encoding="utf-8")
            mapping = build_orf_position_map(str(path))
        self.assertEqual(mapping, {
            "parent": {"a": (1, 9), "b": (11, 19), "c": (21, 29)},
            "other": {"a": (101, 109)},
        })


class OrfIdSchemaTests(unittest.TestCase):
    def test_orf_order_matches_features_in_training_prediction_and_evaluation(self):
        sample = {
            "genome_id": "window", "original_genome_id": "parent",
            "orf_sequences": ["MK", "MA"], "orf_ids": ["b", "a"],
            "bgc_labels": [1, 0], "bgc_class": ["PKS"],
        }
        dictionary_sample = dict(sample, orf_sequences={"b": "MK", "a": "MA"})
        manager = InMemoryFeaturesManager({
            "parent_a": np.array([1., 2., 3.]),
            "parent_b": np.array([4., 5., 6.]),
        })
        expected = torch.tensor([[4., 1.], [5., 2.], [6., 3.]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, features = root / "input.json", root / "features.h5"
            path.write_text(json.dumps([sample, dictionary_sample]))
            with h5py.File(features, "w") as handle:
                handle.create_dataset("ids", data=[b"parent_a", b"parent_b"])
                handle.create_dataset("features", data=manager.features)
            training = RegionBGCDataset(data_file=str(path), features_files=str(features))
            prediction = PredictionDataset(str(path), manager, max_orfs=8)
            evaluation = _InMemoryRegionBGCDataset(str(path), manager, max_orfs=8)
            for i in range(2):
                torch.testing.assert_close(training[i][0], expected)
                torch.testing.assert_close(prediction[i]["features"][:2].T, expected)
                torch.testing.assert_close(evaluation[i][0], expected)
            batch = collate_prediction_batch([prediction[0], prediction[1]])
            self.assertEqual(batch["orf_ids"], [["b", "a"], ["b", "a"]])
            self.assertNotIn("original_orf_ids", batch)
        records = [("parent_b", "MK"), ("parent_a", "MA")]
        self.assertEqual(list(_iter_unique_orfs([sample, dictionary_sample])), records)
        self.assertEqual(list(iter_unique_orfs([sample, dictionary_sample])), records)
        with self.assertRaises(ValueError):
            sample_orf_ids(dict(dictionary_sample, orf_ids=["a", "b"]))

    def test_protein_conversion_emits_only_canonical_orf_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.faa"
            path.write_text(
                ">contig_2 # 11 # 16 # 1 # ID=1_2;partial=00;\nMK\n"
                ">contig_1 # 1 # 6 # 1 # ID=1_1;partial=00;\nMA\n"
            )
            samples = build_samples_from_fasta(str(path), "sample")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["orf_ids"], ["sample_1_2", "sample_1_1"])
        self.assertEqual(list(samples[0]["orf_sequences"]), samples[0]["orf_ids"])
        self.assertNotIn("original_orf_ids", samples[0])


class PredictionThresholdTests(unittest.TestCase):
    def test_missing_configuration_threshold_defaults_to_point_three(self):
        cfg = SimpleNamespace(yaml_cfg={})
        self.assertEqual(_resolve_eval_thresholds(cfg, None, None), (0.3, 0.5))

    def test_prediction_default_ignores_historical_threshold(self):
        class Model(torch.nn.Module):
            def forward(self, features):
                return {}

        def postprocessor(outputs, lengths):
            return [{"segments": torch.tensor([[0, 1], [1, 2]]),
                     "scores": torch.tensor([0.25, 0.35])}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "training" / "bgc_eval"
            history.mkdir(parents=True)
            (history / "best_threshold.json").write_text('{"best_threshold": 0.7}')
            path = root / "input.json"
            path.write_text(json.dumps([{
                "genome_id": "parent", "orf_sequences": {"orf_0": "MA", "orf_1": "MA"},
                "bgc_labels": [0, 0],
            }]))
            cfg = SimpleNamespace(yaml_cfg={"output_dir": str(history.parent)})
            runtime = {"cfg": cfg, "model": Model(), "postprocessor": postprocessor,
                       "device": torch.device("cpu"), "checkpoint_path": "weights.pth"}
            result = run_prediction(
                str(path), ParentFeatures(), str(root / "model.yml"), "weights.pth",
                str(root / "predictions"), device="cpu", runtime=runtime,
            )
            rows = Path(result["orf_txt_path"]).read_text().splitlines()
        self.assertEqual([row.split("\t")[-1] for row in rows[1:]], ["0", "1"])

    def test_configuration_threshold_and_overlap_are_read(self):
        cfg = SimpleNamespace(yaml_cfg={"bgc_eval": {"threshold": 0.3, "overlap_ratio": 0.25}})
        self.assertEqual(_resolve_eval_thresholds(cfg, None, None), (0.3, 0.25))

    def test_explicit_threshold_and_overlap_override_configuration(self):
        cfg = SimpleNamespace(yaml_cfg={
            "bgc_eval": {"threshold": 0.3, "overlap_ratio": 0.5},
        })
        self.assertEqual(_resolve_eval_thresholds(cfg, 0.4, 0.25), (0.4, 0.25))
        self.assertEqual(_resolve_eval_thresholds(cfg, None, None), (0.3, 0.5))



class NucleotideRegionMetricTests(unittest.TestCase):
    def test_nucleotide_lengths_and_inclusive_coordinates(self):
        from src.metrics.regions import compute_bp_region_metrics
        result = compute_bp_region_metrics(
            {'a': [1, 1, 0], 'b': [1]}, {'a': [1, 0, 0], 'b': [1]},
            {'a': [[1, 100], [101, 110], [111, 120]], 'b': [[1, 1]]},
            iou_thresholds=(.75,),
        )['0.75']
        self.assertEqual((result['tp'], result['fp'], result['fn']), (2, 0, 0))
        self.assertAlmostEqual(result['avg_iou'], (100 / 110 + 1) / 2)

    def test_average_iou_is_independent_of_tp_cutoff(self):
        from src.metrics.regions import compute_bp_region_metrics
        result = compute_bp_region_metrics(
            {'a': [1, 1, 0, 1]}, {'a': [1, 0, 0, 1]},
            {'a': [[1, 1], [2, 100], [101, 200], [201, 210]]},
        )
        for values in result.values():
            self.assertEqual((values['tp'], values['fp'], values['fn']), (1, 1, 1))
            self.assertAlmostEqual(values['avg_iou'], .505)

    def test_shuffled_rows_and_cpu_entrypoint_use_saved_pred_column(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from src.evaluation.test_evaluator import evaluate_region_output
        from tools import evaluate_test
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth, predictions, report = root / 'truth.json', root / 'predictions.tsv', root / 'report.json'
            samples = [
                {'genome_id': 'a', 'orf_sequences': {'a_1': 'M', 'a_2': 'M'},
                 'bgc_labels': [1, 1], 'orf_positions': [[1, 10], [11, 100]]},
                {'genome_id': 'b', 'orf_sequences': {'b_1': 'M'},
                 'bgc_labels': [1], 'orf_positions': [[10, 1]]},
            ]
            truth.write_text(json.dumps(samples))
            predictions.write_text('ORF_ID\tProb\tTrue\tPred\n'
                                   'b_1\t0.9\t1\t1\na_2\t0.9\t1\t0\na_1\t0.9\t1\t1\n')
            metrics, counts = evaluate_region_output(prediction_file=str(predictions), truth_file=str(truth))
            self.assertEqual(counts, {'samples': 2, 'orfs': 3})
            self.assertEqual(metrics['0.3']['f1'], .5)
            self.assertAlmostEqual(metrics['0.3']['avg_iou'], .55)
            argv = ['evaluate_test.py', 'regions', '--truth-data', str(truth),
                    '--predictions', str(predictions), '--output-json', str(report)]
            with patch('sys.argv', argv), \
                 patch.object(evaluate_test, '_device', side_effect=AssertionError('No GPU')), \
                 patch.object(evaluate_test, 'extract_features_in_memory', side_effect=AssertionError('No features')), \
                 redirect_stdout(StringIO()):
                evaluate_test.main()
            result = json.loads(report.read_text())
            self.assertEqual(result['coordinate_unit'], 'nucleotide')
            self.assertEqual(result['metrics'], metrics)
            predictions.write_text('ORF_ID\tProb\tTrue\tPred\na_1\t0.9\t1\t1\n')
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                evaluate_region_output(prediction_file=str(predictions), truth_file=str(truth))
            samples[0].pop('orf_positions')
            truth.write_text(json.dumps(samples))
            with self.assertRaisesRegex(ValueError, 'orf_positions'):
                evaluate_region_output(prediction_file=str(predictions), truth_file=str(truth))


class PredictionWorkflowTests(unittest.TestCase):
    def test_cli_controls_and_explicit_checkpoint_selection(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from src.predict import run
        from src.predict.predictor import _find_checkpoint
        arguments = ['--input', 'sample.fa', '--output-dir', 'results']
        args = run.build_parser().parse_args(arguments + ['--threshold', '.4', '--type-threshold', '.65'])
        values = run._build_runtime_args_dict(args)
        self.assertEqual((values['threshold'], values['bgc_type_threshold']), (.4, .65))
        self.assertFalse(values['plot']); self.assertFalse(values['save_json'])
        self.assertNotIn('dataset_json', values)
        for value in ['-.1', '1.1', 'nan', 'inf']:
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                run.build_parser().parse_args(arguments + ['--threshold', value])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'latest.pth').touch()
            cfg = SimpleNamespace(yaml_cfg={'output_dir': str(root)})
            with self.assertRaises(FileNotFoundError):
                _find_checkpoint(cfg, str(root))
            with self.assertRaises(FileNotFoundError):
                _find_checkpoint(cfg, str(root / 'missing.pth'))
            self.assertEqual(_find_checkpoint(cfg, str(root / 'latest.pth')), str(root / 'latest.pth'))

    def test_json_and_plots_only_run_when_requested(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from src.predict import run
        for flags, expect_json, expect_plot in [([], False, False), (['--save-json'], True, False),
                                                (['--verify-sequences'], True, False), (['--plot'], False, True)]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / 'sample.fa'; source.write_text('>a\nATG\n')
                args = run.build_parser().parse_args(['--input', str(source), '--output-dir', str(root / 'out'),
                    '--feature_device', 'cpu', '--predict_device', 'cpu', '--threshold', '.4', '--type-threshold', '.65', *flags])
                manager = SimpleNamespace(get_stats=lambda: {'total_features': 1, 'embedding_dim': 3, 'memory_usage_mb': 1.})
                with patch.object(run, 'prepare_nucleotide_inputs', return_value=(root, root)), \
                     patch.object(run, 'convert_fasta_file_to_dataset', return_value=str(root / 'input.json')), \
                     patch.object(run, 'extract_features_in_memory', return_value=(manager, {})), \
                     patch.object(run, 'run_prediction', return_value={'csv_path': str(root / 'regions.csv')}) as predict, \
                     patch.object(run, 'find_sample_fasta', return_value=source), \
                     patch.object(run, 'build_prediction_json', return_value=str(root / 'regions.json')) as sequence_json, \
                     patch.object(run, 'write_prediction_genbanks', return_value={'region_gbk_paths': []}), \
                     patch.object(run, 'plot_bgc_predictions', return_value=str(root / 'plot.png')) as plot, \
                     redirect_stdout(StringIO()):
                    run._process_single_sample(str(source), 1, 1, run._build_runtime_args_dict(args), None, None, {})
                self.assertEqual(sequence_json.called, expect_json)
                self.assertEqual(plot.called, expect_plot)
                self.assertEqual(predict.call_args.kwargs['threshold'], .4)
                self.assertEqual(predict.call_args.kwargs['bgc_type_threshold'], .65)

    def test_verified_json_is_removed_only_after_pass_and_when_not_requested(self):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        from unittest.mock import patch
        from src.predict import run
        for status, keep in [('PASS', False), ('PASS', True), ('FAIL', False)]:
            with self.subTest(status=status, keep=keep), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / 'sample.fa'; source.write_text('>a\nATG\n')
                sample_dir = root / 'sample'; sample_dir.mkdir()
                sequence_json = sample_dir / 'bgc_regions.json'; sequence_json.write_text('[]')
                item = {'sample_index': 1, 'sample_name': 'sample', 'input_fasta': str(source), 'status': 'success',
                        'prediction_outputs': {'bgc_regions_json': str(sequence_json)}}
                argv = ['predict.py', '--input', str(source), '--output-dir', str(root), '--verify-sequences']
                if keep: argv.append('--save-json')
                with patch('sys.argv', argv), patch.object(run, '_validate_required_paths'), \
                     patch.object(run, '_validate_runtime', return_value=root), \
                     patch.object(run, 'discover_nucleotide_files', return_value=[source]), \
                     patch.object(run, '_process_sample_batch', return_value=[item]), \
                     patch.object(run, 'audit', return_value={'status': status, 'errors': []}), \
                     redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    if status == 'FAIL':
                        with self.assertRaises(SystemExit): run.main()
                    else: run.main()
                self.assertEqual(sequence_json.exists(), keep or status == 'FAIL')
                summary = json.loads((root / 'pipeline_summary.json').read_text())
                self.assertEqual(summary['thresholds'], {'detection': .3, 'classification': .5})
                self.assertEqual(item['sequence_verification'], status)

    def test_thresholds_change_predictions_and_types_without_changing_scores(self):
        import csv
        class Model(torch.nn.Module):
            def forward(self, samples): return {}
        def postprocessor(outputs, lengths):
            return [{'segments': torch.tensor([[0, 1], [1, 2]]), 'scores': torch.tensor([.25, .35]),
                     'bgc_type_probs': torch.tensor([[.6, .4, 0., 0., 0., 0.], [.7, .6, 0., 0., 0., 0.]])}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'input.json'
            source.write_text(json.dumps([{'genome_id': 'parent', 'orf_sequences': {'orf_0': 'MA', 'orf_1': 'MK'},
                                           'orf_positions': [[1, 6], [11, 16]], 'bgc_labels': [0, 0]}]))
            runtime = {'cfg': SimpleNamespace(yaml_cfg={}), 'model': Model(), 'postprocessor': postprocessor,
                       'device': torch.device('cpu'), 'checkpoint_path': 'model.pth'}
            all_scores, predictions, classes, bounds = [], [], [], []
            for index, (threshold, type_threshold) in enumerate([(.3, .5), (.4, .5), (.3, .65), (.3, .8)]):
                output = run_prediction(str(source), ParentFeatures(), str(root / 'model.yml'), 'model.pth',
                    str(root / str(index)), threshold=threshold, bgc_type_threshold=type_threshold, device='cpu', runtime=runtime)
                with Path(output['orf_txt_path']).open() as handle: rows = list(csv.DictReader(handle, delimiter='\t'))
                all_scores.append([row['Prob'] for row in rows]); predictions.append([row['Pred'] for row in rows])
                with Path(output['csv_path']).open() as handle: regions = list(csv.DictReader(handle))
                classes.append([row['class_name'] for row in regions]); bounds.append([(r['start_cds'], r['end_cds']) for r in regions])
            self.assertTrue(all(values == all_scores[0] for values in all_scores))
            self.assertEqual(predictions[0], ['0', '1']); self.assertEqual(predictions[1], ['0', '0'])
            self.assertEqual(classes[0], ['PKS;NRPS']); self.assertEqual(classes[2], ['PKS']); self.assertEqual(classes[3], ['none'])
            self.assertEqual(bounds[0], bounds[2]); self.assertEqual(bounds[0], bounds[3])

if __name__ == "__main__":
    unittest.main()
