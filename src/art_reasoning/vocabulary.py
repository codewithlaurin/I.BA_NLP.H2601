"""A fixed word-to-ID vocabulary built from tokenized training texts."""

from collections.abc import Iterable
from types import MappingProxyType


class Vocabulary:
    """Assign IDs from training tokens; encoding never adds new words.

    Pass only the tokenized training split to the constructor. Use encode()
    with the same vocabulary for training, validation, testing, and inference.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, train_tokens: Iterable[Iterable[str]]) -> None:
        if isinstance(train_tokens, (str, bytes)):
            raise TypeError("train_tokens must contain token sequences, not text")

        token_to_id = {self.PAD_TOKEN: self.PAD_ID, self.UNK_TOKEN: self.UNK_ID}
        for tokens in train_tokens:
            if isinstance(tokens, (str, bytes)):
                raise TypeError("each training text must be a sequence of tokens")
            for token in tokens:
                if not isinstance(token, str):
                    raise TypeError("each token must be a string")
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)

        # Read-only mappings keep word IDs and embedding rows aligned.
        self.token_to_id = MappingProxyType(token_to_id)
        self.id_to_token = tuple(token_to_id)

    def __len__(self) -> int:
        """Return the number of entries, including the two reserved tokens."""
        return len(self.token_to_id)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map ordered word tokens to IDs, using <UNK> for unseen words."""
        if isinstance(tokens, (str, bytes)):
            raise TypeError("tokens must be a sequence of words, not text")
        token_ids: list[int] = []
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError("each token must be a string")
            token_ids.append(self.token_to_id.get(token, self.UNK_ID))
        return token_ids
