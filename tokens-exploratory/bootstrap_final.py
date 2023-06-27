from collections import defaultdict
from pickle import HIGHEST_PROTOCOL, dump, load
from typing import List

import numpy as np
from final import get_counts, get_table
from joblib import Parallel, delayed

RANDOM_SEED = 375
BOOTSTRAP_SIZE = 10000
STATISTIC = "PPMI"
N_JOBS = 16


def get_table_from_corpus(processed_corpus: List[dict]) -> defaultdict:
    counts = get_counts(processed_corpus)
    table = get_table(counts)
    return table


def bootstrap_corpus(
    rng: np.random.Generator, processed_corpus: List[dict]
) -> List[dict]:
    # Miraculously no type casting needed! :)
    chosen = rng.choice(processed_corpus, size=len(processed_corpus), replace=True)
    return chosen


def bootstrap_table(
    rng: np.random.Generator, processed_corpus: List[dict]
) -> defaultdict:
    return get_table_from_corpus(bootstrap_corpus(rng, processed_corpus))


def bootstrap(ground_truth: np.array) -> List[defaultdict]:
    ss = np.random.SeedSequence(RANDOM_SEED)
    child_seeds = ss.spawn(BOOTSTRAP_SIZE)
    # For np.array arguments to a function that exceed max_nbytes, joblib
    # saves the array in question to memory so that it is not copied to each
    # job. We take advantage of that to speed up iteration.
    with Parallel(
        n_jobs=N_JOBS, verbose=100, max_nbytes=ground_truth.nbytes - 1
    ) as parallel:
        streams = parallel(delayed(np.random.default_rng)(seed) for seed in child_seeds)
        bootstraps = parallel(
            delayed(bootstrap_table)(rng, ground_truth) for rng in streams
        )
    return bootstraps


def get_bootstrap_statistic(
    results: List[defaultdict], tokens: List[str], statistic: str
) -> defaultdict(List):
    return {k: [table[k][statistic] for table in results if k in table] for k in tokens}


def get_CI(results: defaultdict(list), confidence: float = 90) -> defaultdict(np.array):
    offset = (100 - confidence) / 2
    return {
        token: np.percentile(results[token], [offset, confidence + offset])
        for token in results
    }


if __name__ == "__main__":
    with open("processed_corpus_list.pickle", "rb") as handle:
        ground_truth_corpus = np.asarray(load(handle))

    with open("exploratory_table.pickle", "rb") as handle:
        ground_truth_table = load(handle)

    ground_truth_tokens = ground_truth_table.keys()

    results = bootstrap(ground_truth_corpus)

    results = get_bootstrap_statistic(
        results, tokens=ground_truth_tokens, statistic=STATISTIC
    )

    CIs = get_CI(results)

    with open("confidence_intervals.pickle", "wb") as handle:
        dump(CIs, handle, protocol=HIGHEST_PROTOCOL)
