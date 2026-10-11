"""Text preprocessing for inputs to fastText embeddings."""

from collections.abc import Iterable
import unicodedata


class FastTextPreprocessor:
    """Apply the same preprocessing to training and inference text.

    Lowercase text, remove punctuation, then collapse whitespace to single
    ordinary spaces. Accents are preserved without Unicode normalization.
    Use tokenize() or tokenize_batch() to obtain word-level token lists.
    """

    def preprocess(self, text: str) -> str:
        """Lowercase text, remove punctuation, and normalize whitespace.

        Punctuation becomes a word boundary, except apostrophes within words,
        which are deleted: ``hello,world`` becomes ``hello world`` and
        ``don't`` becomes ``dont``.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text = text.lower()
        characters: list[str] = []
        for index, character in enumerate(text):
            if unicodedata.category(character).startswith("P"):
                internal_apostrophe = (
                    character in {"'", "’"}
                    and 0 < index < len(text) - 1
                    and text[index - 1].isalnum()
                    and text[index + 1].isalnum()
                )
                if not internal_apostrophe:
                    characters.append(" ")
            else:
                characters.append(character)
        return " ".join("".join(characters).split())

    def tokenize(self, text: str) -> list[str]:
        """Split preprocessed text on whitespace into ordered word-level tokens."""
        return self.preprocess(text).split()

    def tokenize_batch(self, texts: Iterable[str]) -> list[list[str]]:
        """Return one word-token list per text, preserving text and word order."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings, not one string")
        return [self.tokenize(text) for text in texts]

    def preprocess_batch(self, texts: Iterable[str]) -> list[str]:
        """Preprocess multiple texts in their original order."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings, not one string")
        return [self.preprocess(text) for text in texts]
