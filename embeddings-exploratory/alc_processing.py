from pickle import HIGHEST_PROTOCOL, dump, load
from typing import Dict, Iterable, List

import gensim.downloader as api
import numpy as np
from gensim.models import KeyedVectors
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder

N_JOBS = 6


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

        return {"token": word, "X": X, "Y": Y}

    def _get_X(self, toks: List[Dict]) -> np.array:
        encoder = OneHotEncoder(drop="if_binary", sparse_output=False)
        encoder.fit(np.array([[True], [False]]))

        # The conText package creates a matrix of regressors. This is N x V, where V is
        # the #regressors: we know this because their code calls toks_dem@docvars,
        # which checking the Quanteda documentation returns a dataframe with these dimensions.
        return encoder.transform(self._process_is_female(toks))

    def _process_is_female(self, toks: List[Dict]) -> np.array:
        return np.array([[True] if utt["gender"] == "F" else [False] for utt in toks])

    def _get_Y(self, toks: List[Dict]) -> np.array:
        return np.array([self._get_w_v(utt) for utt in toks])

    def _get_w_v(self, utt: Dict) -> np.array:
        assert "u_v" in utt, "Utterance must have u_v metadata"
        return np.matmul(self.transform, utt["u_v"])

    def _tokens_context(self, kw: str) -> Iterable[Dict]:
        return [
            self._process_context_window(i, utt)
            for utt in self.corpus
            for i, tok in enumerate(utt["tokens"])
            if tok == kw
        ]

    def _process_context_window(self, i: int, utt: Dict, window_size=6) -> Dict:
        assert "tokens" in utt, "Must use a valid utterance"
        bounds = window_size // 2
        window = utt["tokens"][i - bounds : i] + utt["tokens"][i + 1 : i + bounds]
        # Based on the principle that the vector for tok ~= this context average
        context_toks = [self._find_vec(tok) for tok in window]
        # TODO: Decide if correct behavior
        u_v = sum(context_toks) / len(context_toks)
        # Necessary as utt is passed by reference
        new = utt.copy()
        new["window"] = context_toks
        new["u_v"] = u_v
        return new

    def _find_vec(self, word: str) -> np.array:
        """Evil hack: if word not found in word_vectors we just return zeros."""
        try:
            return self.embeddings.get_vector(word)
        except KeyError:
            return self._zeros()

    def _zeros(self) -> np.array:
        return np.zeros(shape=self.embeddings.get_vector("a").shape)


def process_wrapper(alc: ALCProcessor, word: str):
    return alc.process(word)


if __name__ == "__main__":
    # GloVe Gigaword 100 dimensions
    word_vectors = api.load("glove-wiki-gigaword-100")
    print("Loaded word vectors")

    matrixfile = "6B.100d.bin"
    A = np.fromfile(matrixfile, dtype=np.float32)
    d = int(np.sqrt(A.shape[0]))
    print("Loaded transform matrix")

    assert (
        d == next(iter(word_vectors)).shape[0]
    ), "induction matrix dimension and word embedding dimension must be the same"
    A = A.reshape(d, d)

    with open("../processed_corpus_list.pickle", "rb") as handle:
        corpus = load(handle)

    print("Loaded corpus")

    alc_processor = ALCProcessor(corpus, word_vectors, A)

    with open("../tokens-exploratory/exploratory_table.pickle", "rb") as handle:
        table = load(handle)

    print("Loaded table")

    tokens = table.keys()

    print("Loaded tokens")

    # word_vectors is thread safe according to gensim docs.
    # The other arguments are np arrays and ALCProcessor uses a lot of numpy
    # operations so hopefully this works more effectively than in bootstrap_final
    output = Parallel(n_jobs=N_JOBS, verbose=100, prefer="threads")(
        delayed(process_wrapper)(alc_processor, token) for token in tokens
    )

    with open("alc_stepone.pickle", "wb") as handle:
        dump(output, handle, protocol=HIGHEST_PROTOCOL)
