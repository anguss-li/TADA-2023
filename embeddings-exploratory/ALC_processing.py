from typing import Dict, Iterable, List

import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder


class ALCProcessor:
    """Performs the same thing as prototype.ipynb."""

    def __init__(
        self, corpus: List[Dict], embeddings: KeyedVectors, transform: np.array
    ) -> None:
        self.corpus = corpus
        self.embeddings = embeddings
        self.transform = transform

    def process(self, word: str) -> Dict:
        toks = self._tokens_context(word)

        X = self._get_X(toks)
        Y = self._get_Y(toks)

        return {"X": X, "Y": Y}

    def _get_X(self, toks: List[Dict]) -> np.array:
        encoder = OneHotEncoder(drop="if_binary", sparse_output=False)
        encoder.fit(np.array([[True], [False]]))

        # The conText package creates a matrix of regressors. This is N x V, where V is
        # the #regressors: we know this because their code calls toks_dem@docvars,
        # which checking the Quanteda documentation returns a dataframe with these dimensions.
        return encoder.transform(self._process_is_female(toks))

    def _process_is_female(self, toks: List[Dict]) -> np.array:
        return np.array([[True] if utt["gender"] == "F" else [False] for utt in toks])

    def _get_Y(self, toks: List[Dict]) -> np.ndarray:
        return np.array([self._get_w_v(utt) for utt in toks])

    def _get_w_v(self, utt: Dict) -> np.array:
        assert "u_v" in utt, "Utterance must have u_v metadata"
        return np.matmul(self.transform, utt["u_v"])

    def _tokens_context(self, kw: str) -> Iterable[Dict]:
        return np.array(
            [
                self._process_context_window(i, utt)
                for utt in self.corpus
                for i, tok in enumerate(utt["tokens"])
                if tok == kw
            ]
        )

    def _process_context_window(self, i: int, utt: Dict, window_size=6) -> Dict:
        assert "tokens" in utt, "Must use a valid utterance"
        bounds = window_size // 2
        window = utt["tokens"][i - bounds : i] + utt["tokens"][i + 1 : i + bounds]
        # Based on the principle that the vector for tok ~= this context average
        context_toks = [self._find_vec(tok) for tok in window]
        u_v = np.true_divide(sum(context_toks), len(context_toks))
        # Necessary as utt is passed by reference
        new = utt.copy()
        new["window"] = context_toks
        new["u_v"] = u_v
        return new

    def _find_vec(self, word: str) -> np.ndarray:
        """Evil hack: if word not found in word_vectors we just return zeros."""
        try:
            return self.embeddings.get_vector(word)
        except KeyError:
            return np.zeros(shape=self.embeddings.get_vector("a").shape)
