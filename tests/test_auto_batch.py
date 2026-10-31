"""Synthetic checks for saved classifier inputs and auto-batch sweep dispatch."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import torch
from torch import nn

import train_auto_batch
import train_classifier
from src.art_reasoning.batching import make_dataloader
from src.art_reasoning.data import prepare_experiment_data
from src.art_reasoning.training import load_trained_classifier, save_inputs
from src.art_reasoning.vocabulary import Vocabulary


class LocalRun:
    def __init__(self):
        self.summary = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def log(self, *args, **kwargs):
        pass


class AutoBatchTests(unittest.TestCase):
    def test_dispatch_reuses_config_and_inputs_without_loading_trainable_weights(self):
        for model_type in ("mean_pool", "rnn"):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "inputs.pt").touch()
                trial = root / "trial"
                trial.mkdir()
                reference = trial / "best.pt"
                config = dict(hidden_dim=256, dropout=0.2, learning_rate=0.001,
                              weight_decay=0.0001, epochs=20, patience=3, min_delta=0.0,
                              oov_policy="fasttext")
                if model_type == "rnn":
                    config.update(rnn_type="gru", rnn_hidden_size=256, rnn_dropout=0.4,
                                  learning_rate=0.003, patience=5, gradient_clip_norm=1.0,
                                  rnn_bidirectional=False, oov_policy="unk")
                # There are deliberately no trainable weights in this fixture.
                torch.save({"config": config, "model_type": model_type,
                            "inputs_file": "../inputs.pt"}, reference)
                backend = train_auto_batch.train_rnn if model_type == "rnn" else train_auto_batch.train_classifier
                with patch.object(backend, "main") as run, contextlib.redirect_stdout(io.StringIO()):
                    train_auto_batch.main([
                        "--reference-checkpoint", str(reference), "--device", "mps", "--seed", "42",
                    ])
                run.assert_called_once()
                argv = run.call_args.args[0]
                self.assertEqual(argv[argv.index("--batch-size") + 1], "auto")
                self.assertEqual(Path(argv[argv.index("--inputs-path") + 1]).resolve(),
                                 (root / "inputs.pt").resolve())
                self.assertEqual(float(argv[argv.index("--learning-rate") + 1]), config["learning_rate"])
                self.assertNotIn("--evaluate-test", argv)
                self.assertNotIn("--reference-checkpoint", argv)
                if model_type == "rnn":
                    self.assertIn("--no-bidirectional", argv)
                    self.assertEqual(argv[argv.index("--rnn-dropout") + 1], "0.4")

    def test_saved_extended_classifier_inputs_use_auto_probe_and_restore(self):
        train = Dataset.from_dict({field: ["known"] * 4 for field in (
            "observation_1", "observation_2", "hypothesis_1", "hypothesis_2"
        )}).add_column("label", [1, 2, 1, 2])
        validation = Dataset.from_dict({
            "observation_1": ["rare", "known"], "observation_2": ["known", "known"],
            "hypothesis_1": ["known", "known"], "hypothesis_2": ["known", "known"], "label": [1, 2],
        })
        dataset = {"train": train, "validation": validation}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vocab = Vocabulary([["known", "rare"]])
            matrix = torch.zeros(4, 300)
            matrix[3] = 2.0
            source = root / "inputs.pt"
            save_inputs(nn.Embedding.from_pretrained(matrix, freeze=True, padding_idx=0), vocab, source)
            before = source.read_bytes()

            def prepare(**kwargs):
                self.assertFalse(kwargs["include_test"])
                self.assertEqual(kwargs["vocabulary"].id_to_token, vocab.id_to_token)
                return prepare_experiment_data(dataset, include_test=False, vocabulary=kwargs["vocabulary"])

            def probe(split, device, **kwargs):
                self.assertIsNotNone(kwargs["probe_step"])
                head, optimizer, loss = kwargs["probe_step"](next(iter(make_dataloader(split, 2))))
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(optimizer.state)
                return 3

            with patch.object(train_classifier, "load_experiment_data", side_effect=prepare), \
                 patch.object(train_classifier, "find_gpu_batch_size", side_effect=probe) as sizing, \
                 patch.object(train_classifier, "load_fasttext_embeddings") as raw_model, \
                 patch("src.art_reasoning.training.start_run", return_value=LocalRun()), \
                 contextlib.redirect_stdout(io.StringIO()):
                train_classifier.main([
                    "--inputs-path", str(source), "--batch-size", "auto", "--device", "cpu",
                    "--hidden-dim", "2", "--epochs", "1", "--patience", "1",
                    "--output-dir", str(root / "out"), "--wandb-mode", "disabled",
                ])
            sizing.assert_called_once()
            raw_model.assert_not_called()
            checkpoint, = (root / "out").glob("*/trial*/best.pt")
            builder, model, restored_vocab = load_trained_classifier(checkpoint)
            self.assertEqual(restored_vocab.id_to_token, vocab.id_to_token)
            self.assertTrue(torch.equal(builder.embedding.weight, matrix))
            config = json.loads((checkpoint.parent / "config.json").read_text())
            self.assertEqual(config["batch_size"], 3)
            self.assertTrue(config["batch_size_automatic"])
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
