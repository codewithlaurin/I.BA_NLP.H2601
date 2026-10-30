"""Synthetic correctness checks; no real FastText binary or ART split is loaded."""

from dataclasses import asdict
import contextlib
import io
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import torch
from torch import nn

import compare_fasttext_oov
from src.art_reasoning.embeddings import extend_fasttext_embeddings
from src.art_reasoning.models.classifier import Classifier
from src.art_reasoning.training import RNNTrainingConfig, TrainingConfig, build_rnn_model, save_inputs
from src.art_reasoning.vocabulary import Vocabulary


class FakeFastText:
    def get_dimension(self):
        return 300

    def get_word_vector(self, word):
        return np.full(300, 2.0 if word == "rare" else 3.0, dtype=np.float32)


class FastTextExtensionTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = Vocabulary([["known"]])
        self.embedding = nn.Embedding.from_pretrained(
            torch.zeros(3, 300), freeze=True, padding_idx=0
        )
        # Use a nonzero UNK row to detect accidental recomputation or replacement.
        self.embedding.weight[1].fill_(0.25)

    def test_extension_preserves_ids_vectors_and_original_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fake.bin"
            path.touch()
            with patch("src.art_reasoning.embeddings.fasttext.load_model", return_value=FakeFastText()):
                extended, vocabulary = extend_fasttext_embeddings(
                    self.embedding, self.vocabulary, [["known", "rare", "rare"]], path
                )
        self.assertEqual(len(vocabulary), 4)
        self.assertEqual(vocabulary.encode(["known", "rare"]), [2, 3])
        self.assertEqual(self.vocabulary.encode(["rare"]), [1])
        self.assertTrue(torch.equal(extended.weight[:3], self.embedding.weight))
        self.assertTrue(torch.equal(extended.weight[3], torch.full((300,), 2.0)))
        self.assertFalse(extended.weight.requires_grad)
        self.assertEqual(extended.padding_idx, 0)

    def test_cached_words_do_not_reload_fasttext(self):
        with patch("src.art_reasoning.embeddings.fasttext.load_model") as loader:
            embedding, vocabulary = extend_fasttext_embeddings(
                self.embedding, self.vocabulary, [["known"]], "unused.bin"
            )
        loader.assert_not_called()
        self.assertIs(embedding, self.embedding)
        self.assertIs(vocabulary, self.vocabulary)

    def test_second_extension_keeps_previous_cached_vector(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fake.bin"
            path.touch()
            fake = FakeFastText()
            with patch("src.art_reasoning.embeddings.fasttext.load_model", return_value=fake), \
                 patch.object(fake, "get_word_vector", wraps=fake.get_word_vector) as lookup:
                first, vocab = extend_fasttext_embeddings(
                    self.embedding, self.vocabulary, [["rare"]], path
                )
                second, vocab = extend_fasttext_embeddings(first, vocab, [["rare", "new"]], path)
        self.assertEqual([call.args[0] for call in lookup.call_args_list], ["rare", "new"])
        self.assertEqual(vocab.encode(["known", "rare", "new"]), [2, 3, 4])
        self.assertTrue(torch.equal(first.weight, second.weight[:4]))

    def test_rejects_trainable_embeddings_and_invalid_token_sequences(self):
        self.embedding.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            extend_fasttext_embeddings(self.embedding, self.vocabulary, [["rare"]], "unused.bin")
        with self.assertRaises(TypeError):
            self.vocabulary.extended("rare")


class ComparisonCommandTests(unittest.TestCase):
    def test_baseline_and_rnn_compare_same_weights_on_each_synthetic_split(self):
        dataset = Dataset.from_dict({
            "observation_1": ["rare", "known"],
            "observation_2": ["known", "known"],
            "hypothesis_1": ["known", "known"],
            "hypothesis_2": ["known", "known"],
            "label": [2, 1],
        })
        for model_type, split in product(("mean_pool", "rnn"), ("validation", "test")):
            with self.subTest(model_type=model_type, split=split), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                vocabulary = Vocabulary([["known"]])
                embedding = nn.Embedding.from_pretrained(
                    torch.zeros(3, 300), freeze=True, padding_idx=0
                )
                save_inputs(embedding, vocabulary, root / "inputs.pt")
                if model_type == "mean_pool":
                    config = TrainingConfig(hidden_dim=1, dropout=0.0)
                    model = Classifier(hidden_dim=1, dropout=0.0)
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.zero_()
                        model.layers[0].weight[0, 0] = 1.0
                        model.layers[3].weight[0, 0] = -1.0
                        model.layers[3].weight[1, 0] = 1.0
                        model.layers[3].bias[0] = 0.1
                    state = {"classifier_state": model.state_dict()}
                else:
                    config = RNNTrainingConfig(hidden_dim=2, rnn_hidden_size=2, dropout=0.0)
                    model = build_rnn_model(embedding, config)
                    state = {"model_state": {key: value for key, value in model.state_dict().items()
                                             if key != "encoder.embedding.weight"}}
                trial = root / "trial"
                trial.mkdir()
                checkpoint = trial / "best.pt"
                torch.save({
                    "model_type": model_type, "checkpoint_format_version": 2 if model_type == "rnn" else 1,
                    "config": asdict(config), "inputs_file": "../inputs.pt", "epoch": 1,
                    "metrics": {"validation/loss": 1.0, "validation/accuracy": 0.5},
                    **state,
                }, checkpoint)
                before = compare_fasttext_oov.sha256(checkpoint)
                fasttext_path = root / "fake.bin"
                fasttext_path.touch()
                output = root / "results"
                with patch.object(compare_fasttext_oov, "load_dataset", return_value=dataset) as loader, \
                     patch("src.art_reasoning.embeddings.fasttext.load_model", return_value=FakeFastText()), \
                     patch.object(compare_fasttext_oov, "start_run") as tracking, \
                     contextlib.redirect_stdout(io.StringIO()):
                    argv = [
                        str(checkpoint), "--fasttext-model", str(fasttext_path),
                        "--output-dir", str(output), "--batch-size", "1",
                    ]
                    if split == "test":
                        argv += ["--split", "test"]
                    compare_fasttext_oov.main(argv)
                loader.assert_called_once_with("amarfurt/art-nlp-hs26", split=split)
                tracking.assert_not_called()
                summary_path, = output.glob("*/summary.json")
                result = json.loads(summary_path.read_text())
                predictions = [json.loads(line) for line in
                               (summary_path.parent / "predictions.jsonl").read_text().splitlines()]
                self.assertEqual(result["phase"], split)
                self.assertEqual(result["added_word_types"], 1)
                self.assertEqual(result["original_oov_tokens"], 1)
                self.assertEqual(result["expanded_oov_tokens"], 0)
                self.assertEqual(result["examples_with_oov"], 1)
                self.assertEqual(result["unk"]["examples"], 2)
                self.assertEqual(result["fasttext"]["examples"], 2)
                self.assertEqual(predictions[1]["unk_prediction"], predictions[1]["fasttext_prediction"])
                self.assertEqual(compare_fasttext_oov.sha256(checkpoint), before)
                if model_type == "mean_pool":
                    self.assertEqual(result["unk"]["correct"], 1)
                    self.assertEqual(result["fasttext"]["correct"], 2)
                    self.assertEqual(result["wrong_to_correct"], 1)
                    self.assertEqual(result["correct_to_wrong"], 0)

    def test_validation_cache_option_cannot_select_test(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            compare_fasttext_oov.main([
                "unused.pt", "--fasttext-model", "unused.bin", "--split", "test",
                "--validation-arrow", "unused.arrow",
            ])


if __name__ == "__main__":
    unittest.main()
