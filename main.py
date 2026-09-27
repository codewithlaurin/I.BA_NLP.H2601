from datasets import load_dataset

def main():
    train = load_dataset("amarfurt/art-nlp-hs26", split="train")
    valid = load_dataset("amarfurt/art-nlp-hs26", split="validation")
    test = load_dataset("amarfurt/art-nlp-hs26", split="test")
    print(len(train), len(valid), len(test))


if __name__ == "__main__":
    main()
