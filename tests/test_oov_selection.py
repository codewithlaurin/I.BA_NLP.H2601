"""Matched-policy training checks using only synthetic examples and vectors."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import torch
from torch import nn

import train_oov_policy
from src.art_reasoning.batching import make_dataloader
from src.art_reasoning.data import _encode_split, _tokenize_split
from src.art_reasoning.preprocessing import FastTextPreprocessor
from src.art_reasoning.training import load_trained_classifier, load_trained_rnn, run_epoch, save_inputs
from src.art_reasoning.vocabulary import Vocabulary


class FakeFastText:
    def get_dimension(self):
        return 300

    def get_word_vector(self, word):
        return np.full(300, 2.0, dtype=np.float32)


class LocalRun:
    def __init__(self):
        self.summary = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def log(self, *args, **kwargs):
        pass


class OOVSelectionTests(unittest.TestCase):
    def test_policies_have_identical_training_and_restorable_selected_checkpoints(self):
        train = Dataset.from_dict({
            field: ["known"] * 8 for field in (
                "observation_1", "observation_2", "hypothesis_1", "hypothesis_2"
            )
        }).add_column("label", [1, 2] * 4)
        validation = Dataset.from_dict({
            "observation_1": ["rare", "known"], "observation_2": ["known", "known"],
            "hypothesis_1": ["known", "known"], "hypothesis_2": ["known", "known"],
            "label": [2, 1],
        })
        dataset = {"train": train, "validation": validation}
        for family in ("mean_pool", "gru"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "inputs.pt"
                save_inputs(nn.Embedding.from_pretrained(torch.zeros(3, 300), freeze=True, padding_idx=0),
                            Vocabulary([["known"]]), source)
                fake_bin = root / "fake.bin"
                fake_bin.touch()
                source_before = source.read_bytes()
                checkpoints = {}
                curves = {}
                for policy in ("unk", "fasttext"):
                    with patch.object(train_oov_policy, "load_art_dataset", return_value=dataset) as loader, \
                         patch("src.art_reasoning.embeddings.fasttext.load_model", return_value=FakeFastText()) as ft, \
                         patch("src.art_reasoning.training.start_run", return_value=LocalRun()), \
                         contextlib.redirect_stdout(io.StringIO()):
                        train_oov_policy.main([
                            "--model-family", family, "--oov-policy", policy,
                            "--inputs-path", str(source), "--fasttext-model", str(fake_bin),
                            "--epochs", "3", "--baseline-patience", "5", "--rnn-patience", "5",
                            "--batch-size", "4", "--device", "cpu", "--seed", "42",
                            "--wandb-mode", "disabled", "--no-wandb-log-model",
                            "--output-dir", str(root / policy),
                        ])
                    loader.assert_called_once_with(cache_dir=None, include_test=False)
                    if policy == "unk":
                        ft.assert_not_called()
                    else:
                        ft.assert_called_once()
                    config_path, = (root / policy).glob("*/trial/config.json")
                    config = json.loads(config_path.read_text())
                    self.assertEqual(set(config["split_sizes"]), {"train", "validation"})
                    self.assertEqual(config["validation_oov_tokens"], 1 if policy == "unk" else 0)
                    self.assertEqual(config["oov_policy"], policy)
                    trial = config_path.parent
                    curves[policy] = [json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()]
                    checkpoints[policy] = torch.load(trial / "last.pt", weights_only=True, map_location="cpu")
                    selected = torch.load(trial / "best.pt", weights_only=True, map_location="cpu")
                    minimum = min(curves[policy], key=lambda row: row["validation/loss"])
                    self.assertEqual(selected["epoch"], minimum["epoch"])
                    if family == "gru":
                        model, vocab = load_trained_rnn(trial / "best.pt")
                        builder = None
                    else:
                        builder, model, vocab = load_trained_classifier(trial / "best.pt")
                    tokens, labels = _tokenize_split(validation, FastTextPreprocessor())
                    restored = run_epoch(builder, model, make_dataloader(_encode_split(tokens, labels, vocab), 4), "cpu")
                    self.assertAlmostEqual(restored["loss"], minimum["validation/loss"], places=6)
                    self.assertEqual(restored["accuracy"], minimum["validation/accuracy"])
                self.assertEqual(source.read_bytes(), source_before)
                self.assertEqual(len(curves["unk"]), 3)
                self.assertEqual(len(curves["fasttext"]), 3)
                for original, expanded in zip(curves["unk"], curves["fasttext"]):
                    self.assertEqual(original["train/loss"], expanded["train/loss"])
                    self.assertEqual(original["train/accuracy"], expanded["train/accuracy"])
                key = "model_state" if family == "gru" else "classifier_state"
                for name, tensor in checkpoints["unk"][key].items():
                    self.assertTrue(torch.equal(tensor, checkpoints["fasttext"][key][name]), name)


if __name__ == "__main__":
    unittest.main()
