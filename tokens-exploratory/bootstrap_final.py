from collections import defaultdict
from pickle import load, dump, HIGHEST_PROTOCOL
from typing import List, TypeVar

import numpy as np
from final import get_corpus, get_counts, get_table, process_corpus
from joblib import Parallel, delayed

RANDOM_SEED = 375
BOOTSTRAP_SIZE = 10
STATISTIC = "PPMI"
N_JOBS = 16
Comparable = TypeVar("Comparable")


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


def bootstrap_table(seed: int, processed_corpus: List[dict]) -> defaultdict:
    rng = np.random.default_rng(seed)
    return get_table_from_corpus(bootstrap_corpus(rng, processed_corpus))


def bootstrap(
    ground_truth: List[dict],
) -> List[defaultdict]:
    return Parallel(n_jobs=N_JOBS, verbose=100)(
        delayed(bootstrap_table)(seed, ground_truth)
        for seed in range(RANDOM_SEED, RANDOM_SEED + BOOTSTRAP_SIZE)
    )


def get_bootstrap_statistic(
    results: List[defaultdict], tokens: List[str], statistic: str
) -> defaultdict:
    return {k: [table[k][statistic] for table in results if k in table] for k in tokens}


def get_p_values(
    results: defaultdict, ground_truth: defaultdict, statistic: str
) -> defaultdict:
    p_values = defaultdict(float)
    for token in results:
        if ground_truth[token][statistic] == None:
            continue

        p = significance_test(results[token], ground_truth[token][statistic])
        p_values[token] = p
    return p_values


def significance_test(results: List[Comparable], ground_truth: Comparable) -> float:
    return sum(1 if x >= ground_truth else 0 for x in results) / len(results)


if __name__ == "__main__":
    with open("processed_corpus_list.pickle", "rb") as handle:
        ground_truth_corpus = load(handle)

    ground_truth_table = get_table_from_corpus(ground_truth_corpus)
    ground_truth_tokens = ground_truth_table.keys()

    results = bootstrap(ground_truth_corpus)
    
    print(results[5:8])
    results = get_bootstrap_statistic(
        results, tokens=ground_truth_tokens, statistic=STATISTIC
    )

    # We quickly filter out any token where a bootstrap iteration leads to a
    # NaN value for STATISTIC
    results = {
        token: results[token] for token in results if not (None in results[token])
    }

    p_values = get_p_values(results, ground_truth_table, STATISTIC)
    print(p_values)
    
    # with open("p_values.pickle", "wb") as handle:
    #     dump(p_values, handle, protocol=HIGHEST_PROTOCOL)
