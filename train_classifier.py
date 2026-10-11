"""Prepare all ART texts as token-ID matrices for classifier experiments."""

from src.art_reasoning.data import TEXT_FIELDS, load_experiment_data


def main():
    data = load_experiment_data()
    print(f"Vocabulary size (including <PAD> and <UNK>): {len(data.vocabulary)}")
    print(f"Text rows within each example: {', '.join(TEXT_FIELDS)}")
    for name in ("train", "validation", "test"):
        split = getattr(data, name)
        print(
            f"{name}: {len(split.labels)} examples, "
            f"ID matrix {tuple(split.token_ids.shape)}, "
            f"lengths {tuple(split.lengths.shape)}, "
            f"labels {tuple(split.labels.shape)}"
        )
    print("First training example's token IDs (first 10 columns):")
    print(data.train.token_ids[: len(TEXT_FIELDS), :10])


if __name__ == "__main__":
    main()
