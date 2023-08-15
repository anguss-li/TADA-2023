from collections import Counter, defaultdict
from pickle import HIGHEST_PROTOCOL, dump, load
from typing import List

import numpy as np
from convokit import Corpus

# Some notation:
#   - T is a variable in a causal inference problem.
#   - We assume T is a binary variable s.t. T = {T_A, T_B}
T = "gender"
T_A, T_B = "M", "F"


def get_corpus() -> Corpus:
    with open("../processed_corpus_list.pickle", "rb") as handle:
        corpus = load(handle)

    return corpus


def get_counts(processed_corpus: List[dict]) -> "defaultdict[Counter]":
    """Our goal is to build a table with each word token in the vocabulary for
    the Supreme Court corpus, listing each token's number of male and female speakers.

    Args:
        corpus (Corpus): Supreme Court corpus. Assumed to be tokenized, cleaned etc.

    Returns:
        defaultdict[Counter]:
            keys: tokens in the vocabulary
            values: Counter dictionaries
                keys: str, gender signals
                values: how many speakers of given gender have said the token
    """
    counts = defaultdict(Counter)

    for utt in processed_corpus:
        for token in utt["tokens"]:
            counts[token][utt[T]] += 1

    return counts


def get_PPMI_table(
    counts: defaultdict(Counter), smoothing_factor: float = 1e-10
) -> defaultdict(defaultdict):
    """_summary_

    Args:
        counts (defaultdict[Counter]):
            keys: tokens in the vocabulary
            values: Counter dictionaries
                keys: str, category out of {T_A, T_B}
                values: how many speakers of given category have said the token

    Returns:
        defaultdict[defaultdict]:
            keys: tokens in the vocabulary
            values: default dicts
                keys: str, counts and ratios for speakers in {T_A, T_B}
                values: int/floats
    """
    # Outer layer defaultdict returns an inner defaultdict when key not present.
    # This inner defaultdict returns None when a key to it is not present.
    table = defaultdict(defaultdict)

    # The total number of times a token is used, summed for all tokens in V
    total_vocab_usage = 0
    # The total number of times a token is used by a female speaker, summed for all tokens in V
    T_B_total_vocab_usage = 0

    for token in counts:
        table[token][f"{T_A} count"] = counts[token][T_A]
        table[token][f"{T_B} count"] = counts[token][T_B]
        table[token][f"{T} total"] = table[token][f"{T_A} count"] + table[token][f"{T_B} count"]

        table[token][f"{T_A} ratio"] = table[token][f"{T_A} count"] / table[token]["total"]
        table[token][f"{T_B} ratio"] = table[token][f"{T_B} count"] / table[token]["total"]
        table[token][f"{T_B} - {T_A}"] = table[token][f"{T_B} ratio"] - table[token][f"{T_A} ratio"]

        total_vocab_usage += table[token]["total"]
        T_B_total_vocab_usage += table[token][f"{T_B} count"]

    # The proportion of all token usage in the corpus done by female speakers
    p_j = T_B_total_vocab_usage / total_vocab_usage

    for token in table:
        # The proportion of all token usage in the corpus taken up by this specific token
        p_i = table[token]["total"] / total_vocab_usage
        # The proportion of all token usage in the corpus taken up by this specific token,
        # and where the speaker is T_B
        p_ij = table[token][f"{T_B} count"] / total_vocab_usage
        # The positive pointwise mutual information (PMI) between:
        #   i: the event this token is used
        #   j: the event a T_B speaker speaks (uses a word in V)
        PMI = np.log2(
            (p_ij + smoothing_factor)
            / ((p_i + smoothing_factor) * (p_j + smoothing_factor))
        )
        PPMI = max(PMI, 0)
        table[token]["PPMI"] = PPMI

    return table


if __name__ == "__main__":
    corpus = get_corpus()
    counts = get_counts(corpus)
    table = get_PPMI_table(counts)
    with open("exploratory_table.pickle", "wb") as handle:
        dump(table, handle, protocol=HIGHEST_PROTOCOL)
