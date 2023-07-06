import gensim.downloader as api
from pickle import load, dump, HIGHEST_PROTOCOL
from joblib import Parallel, delayed

from ALC_processing import *  # imports all functions from chatbot.py

N_JOBS = 6


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
