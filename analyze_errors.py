"""Compare saved ART predictions of several models and write an error-analysis report.

Inputs are the prediction files the evaluation scripts already write:
evaluate_transformer.py -> <run>/<split>_predictions.jsonl, evaluate_llm.py -> <run>/responses.jsonl.
Each input is given as NAME=PATH, for example pretrained=outputs/project2/<run>/validation_predictions.jsonl.
"""

import argparse
from collections import Counter
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import random
import re

WORD = re.compile(r"[a-z0-9']+")
CONFIDENCE_BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
EXAMPLE_SEED = 0


def read_predictions(path: Path) -> list[dict]:
    with path.open() as stream:
        records = [json.loads(line) for line in stream]
    if [record["example_index"] for record in records] != list(range(len(records))):
        raise ValueError(f"{path}: examples must be complete and in dataset order")
    return records


def parse_model_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def surface_cues(example: dict) -> dict[str, int | None]:
    """Hypothesis (1 or 2) that a simple surface cue would choose; None on a tie."""
    lengths = [len(example[f"hypothesis_{choice}"].split()) for choice in (1, 2)]
    observation_words = words(example["observation_1"]) | words(example["observation_2"])
    overlaps = [len(words(example[f"hypothesis_{choice}"]) & observation_words) for choice in (1, 2)]

    def larger(values: list[int]) -> int | None:
        return None if values[0] == values[1] else 1 + values.index(max(values))

    return {"longer hypothesis": larger(lengths), "more word overlap with observations": larger(overlaps)}


def confidence(record: dict) -> float | None:
    """Probability of the predicted hypothesis; LLM replies have none."""
    if "probabilities" not in record:
        return None
    return max(record["probabilities"])


def percent(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "-"


def model_metrics(records: list[dict]) -> dict:
    confusion = {label: Counter() for label in (1, 2)}
    for record in records:
        confusion[record["label"]][record["predicted_label"] or "invalid"] += 1
    count = len(records)
    correct = sum(confusion[label][label] for label in (1, 2))
    invalid = sum(confusion[label]["invalid"] for label in (1, 2))
    predicted_first = sum(confusion[label][1] for label in (1, 2))
    return {
        "examples": count, "correct": correct, "accuracy": correct / count,
        "invalid": invalid, "predicted_hypothesis_1": predicted_first,
        "gold_hypothesis_1": sum(confusion[1].values()),
        "confusion_matrix": {f"true {label}": {str(column): confusion[label][column]
                                               for column in (1, 2, "invalid")}
                             for label in (1, 2)},
    }


def agreement(first: list[dict], second: list[dict]) -> dict[str, int]:
    counts = Counter()
    for a, b in zip(first, second, strict=True):
        a_correct = a["predicted_label"] == a["label"]
        b_correct = b["predicted_label"] == b["label"]
        counts[{(True, True): "both correct", (True, False): "only first correct",
                (False, True): "only second correct", (False, False): "both wrong"}[a_correct, b_correct]] += 1
    return dict(counts)


def cue_table(examples: list[dict], models: dict[str, list[dict]]) -> list[dict]:
    """How often each cue is right, and each model's accuracy when it is right or wrong."""
    rows = []
    cues = [surface_cues(example) for example in examples]
    for cue_name in cues[0]:
        decided = [index for index, cue in enumerate(cues) if cue[cue_name] is not None]
        cue_right = [index for index in decided if cues[index][cue_name] == examples[index]["label"]]
        cue_wrong = [index for index in decided if index not in set(cue_right)]
        row = {"cue": cue_name, "decided": len(decided), "ties": len(examples) - len(decided),
               "cue_accuracy": len(cue_right) / len(decided), "models": {}}
        for name, records in models.items():
            def accuracy(indices: list[int]) -> float:
                return sum(records[i]["predicted_label"] == records[i]["label"] for i in indices) / len(indices)
            follows = sum(records[i]["predicted_label"] == cues[i][cue_name] for i in decided)
            row["models"][name] = {
                "follows_cue": follows / len(decided),
                "accuracy_when_cue_right": accuracy(cue_right),
                "accuracy_when_cue_wrong": accuracy(cue_wrong),
            }
        rows.append(row)
    return rows


def confidence_table(records: list[dict]) -> list[dict]:
    rows = []
    for low, high in zip(CONFIDENCE_BINS, CONFIDENCE_BINS[1:]):
        members = [r for r in records
                   if low <= confidence(r) < high or (high == 1.0 and confidence(r) == 1.0)]
        correct = sum(r["predicted_label"] == r["label"] for r in members)
        rows.append({"bin": f"{low:.1f}-{high:.1f}", "examples": len(members),
                     "accuracy": correct / len(members) if members else None})
    return rows


def describe_example(index: int, examples: list[dict], models: dict[str, list[dict]]) -> str:
    example = examples[index]
    predictions = []
    for name, records in models.items():
        record = records[index]
        answer = record["predicted_label"] or f"invalid ({record.get('raw_response', '')!r})"
        probability = confidence(record)
        predictions.append(f"{name} {answer}" + (f" (p={probability:.2f})" if probability else ""))
    return "\n".join([
        f"- **Example {index}** (gold: hypothesis {example['label']}; predicted: {', '.join(predictions)})",
        f"  - Observation 1: {example['observation_1']}",
        f"  - Hypothesis 1: {example['hypothesis_1']}",
        f"  - Hypothesis 2: {example['hypothesis_2']}",
        f"  - Observation 2: {example['observation_2']}",
    ])


def select_examples(examples: list[dict], models: dict[str, list[dict]], per_group: int) -> dict[str, list[int]]:
    """Fixed-seed samples, so the report does not show cherry-picked examples."""
    sampler = random.Random(EXAMPLE_SEED)
    correct = {name: [r["predicted_label"] == r["label"] for r in records] for name, records in models.items()}
    indices = range(len(examples))
    groups = {"all models wrong": [i for i in indices if not any(c[i] for c in correct.values())]}
    for first, second in combinations(models, 2):
        groups[f"only {first} correct (vs {second})"] = [
            i for i in indices if correct[first][i] and not correct[second][i]]
        groups[f"only {second} correct (vs {first})"] = [
            i for i in indices if correct[second][i] and not correct[first][i]]
    selected = {group: sorted(sampler.sample(members, min(per_group, len(members))))
                for group, members in groups.items()}
    for name, records in models.items():
        if "probabilities" in records[0]:
            wrong = [i for i in indices if not correct[name][i]]
            selected[f"most confident {name} errors"] = sorted(
                wrong, key=lambda i: -confidence(records[i]))[:per_group]
    return selected


def write_report(split: str, sources: dict[str, Path], metrics: dict, agreements: dict,
                 cues: list[dict], confidences: dict, examples: list[dict],
                 models: dict[str, list[dict]], selected: dict[str, list[int]]) -> str:
    names = list(models)
    lines = [f"# Error analysis: {split} split", "", "Sources:", ""]
    lines += [f"- {name}: `{path}`" for name, path in sources.items()]
    lines += ["", "## Accuracy and confusion matrices", "",
              "| Model | Accuracy | Correct | Invalid | Predicted hypothesis 1 | Gold hypothesis 1 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in names:
        m = metrics[name]
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['correct']}/{m['examples']} | {m['invalid']} | "
                     f"{percent(m['predicted_hypothesis_1'], m['examples'])} | "
                     f"{percent(m['gold_hypothesis_1'], m['examples'])} |")
    for name in names:
        matrix = metrics[name]["confusion_matrix"]
        lines += ["", f"**{name}** (rows: gold, columns: predicted)", "",
                  "| | predicted 1 | predicted 2 | invalid |", "| --- | --- | --- | --- |"]
        lines += [f"| {row} | {cells['1']} | {cells['2']} | {cells['invalid']} |" for row, cells in matrix.items()]
    lines += ["", "## Agreement between models", "",
              "| First | Second | Both correct | Only first | Only second | Both wrong |",
              "| --- | --- | --- | --- | --- | --- |"]
    for (first, second), counts in agreements.items():
        lines.append(f"| {first} | {second} | {counts.get('both correct', 0)} | "
                     f"{counts.get('only first correct', 0)} | {counts.get('only second correct', 0)} | "
                     f"{counts.get('both wrong', 0)} |")
    lines += ["", "## Surface cues", "",
              "A cue picks the longer hypothesis (words) or the one sharing more lower-cased words with the "
              "two observations; ties are excluded. Model accuracy is split by whether the cue picks the gold hypothesis.",
              "", "| Cue | Cue accuracy | Ties | Model | Follows cue | Accuracy, cue right | Accuracy, cue wrong |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in cues:
        for name, values in row["models"].items():
            lines.append(f"| {row['cue']} | {row['cue_accuracy']:.3f} | {row['ties']} | {name} | "
                         f"{values['follows_cue']:.1%} | {values['accuracy_when_cue_right']:.3f} | "
                         f"{values['accuracy_when_cue_wrong']:.3f} |")
    if confidences:
        lines += ["", "## Accuracy by confidence (probability of the predicted hypothesis)", "",
                  "| Model | " + " | ".join(row["bin"] for row in next(iter(confidences.values()))) + " |",
                  "| --- |" + " --- |" * (len(CONFIDENCE_BINS) - 1)]
        for name, rows in confidences.items():
            cells = [f"{row['accuracy']:.2f} (n={row['examples']})" if row["examples"] else "-" for row in rows]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Examples", "",
              f"Random samples with seed {EXAMPLE_SEED}, except the most confident errors. Story order: "
              "observation 1, hypothesis, observation 2."]
    for group, indices in selected.items():
        lines += ["", f"### {group}", ""]
        lines += [describe_example(i, examples, models) for i in indices] or ["None."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=parse_model_argument, metavar="NAME=PATH")
    parser.add_argument("--examples-per-group", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    sources = dict(args.models)
    if len(sources) != len(args.models):
        parser.error("model names must be unique")
    models = {name: read_predictions(path) for name, path in sources.items()}

    examples = next(iter(models.values()))
    fields = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2", "label")
    for name, records in models.items():
        if [[r[f] for f in fields] for r in records] != [[e[f] for f in fields] for e in examples]:
            raise ValueError(f"{name}: examples differ from {next(iter(models))}; use the same full split")
    split_sizes = {1000: "validation", 1532: "test", 16801: "train"}
    split = split_sizes.get(len(examples), f"{len(examples)} examples")

    metrics = {name: model_metrics(records) for name, records in models.items()}
    agreements = {(first, second): agreement(models[first], models[second])
                  for first, second in combinations(models, 2)}
    cues = cue_table(examples, models)
    confidences = {name: confidence_table(records) for name, records in models.items()
                   if "probabilities" in records[0]}
    selected = select_examples(examples, models, args.examples_per_group)

    output_dir = args.output_dir or Path("outputs/project2/error-analysis") / (
        f"{split.replace(' ', '-')}_{datetime.now():%Y%m%d-%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "split": split, "sources": {name: str(path) for name, path in sources.items()},
        "metrics": metrics,
        "agreement": {f"{first} vs {second}": counts for (first, second), counts in agreements.items()},
        "surface_cues": cues, "accuracy_by_confidence": confidences,
        "example_seed": EXAMPLE_SEED, "examples": selected,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = write_report(split, sources, metrics, agreements, cues, confidences, examples, models, selected)
    (output_dir / "report.md").write_text(report)
    print(report)
    print(f"Written to {output_dir}")


if __name__ == "__main__":
    main()
