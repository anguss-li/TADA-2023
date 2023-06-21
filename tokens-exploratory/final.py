from collections import Counter, defaultdict
from pickle import HIGHEST_PROTOCOL, dump
from typing import List

import numpy as np
from convokit import Corpus, Utterance


def get_corpus() -> Corpus:
    corpus = Corpus(filename="../supreme_full_processed_lem")

    # Filter out speakers without gender signal
    corpus = corpus.filter_utterances_by(
        lambda utt: utt.get_speaker().retrieve_meta("gender_signal") != None
    )

    return corpus


def get_counts(corpus: Corpus) -> "defaultdict[Counter]":
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

    for utt in corpus.iter_utterances():
        tokens = utt.retrieve_meta("lem-tokens")
        gender = utt.get_speaker().retrieve_meta("gender_signal")
        for token in tokens:
            counts[token][gender] += 1

    return counts


def get_table(counts: "defaultdict[Counter]") -> "defaultdict[defaultdict]":
    """_summary_

    Args:
        counts (defaultdict[Counter]):
            keys: tokens in the vocabulary
            values: Counter dictionaries
                keys: str, gender signals
                values: how many speakers of given gender have said the token

    Returns:
        defaultdict[defaultdict]:
            keys: tokens in the vocabulary
            values: default dicts
                keys: str, counts and ratios for male and female speakers
                values: int/floats
    """
    # Add percentages in a second run because this is a bit neater.
    table = defaultdict(defaultdict)

    for token in counts:
        table[token]["M count"] = counts[token]["M"]
        table[token]["F count"] = counts[token]["F"]
        table[token]["total"] = table[token]["M count"] + table[token]["F count"]

        table[token]["M ratio"] = table[token]["M count"] / table[token]["total"]
        table[token]["F ratio"] = table[token]["F count"] / table[token]["total"]
        table[token]["F - M"] = table[token]["F ratio"] - table[token]["M ratio"]

    # The total number of times a token is used, summed for all tokens in V
    total_vocab_usage = sum(table[token]["total"] for token in table)
    # The total number of times a token is used by a female speaker, summed for all tokens in V
    female_total_vocab_usage = sum(table[token]["F count"] for token in table)
    # The proportion of all token usage in the corpus done by female speakers
    p_j = female_total_vocab_usage / total_vocab_usage

    for token in table:
        # The proportion of all token usage in the corpus taken up by this specific token
        p_i = table[token]["total"] / total_vocab_usage
        # The proportion of all token usage in the corpus taken up by this specific token,
        # and where the speaker is female
        p_ij = table[token]["F count"] / total_vocab_usage
        # The positive pointwise mutual information (PMI) between:
        #   i: the event this token is used
        #   j: the event a female speaker speaks (uses a word in V)
        PPMI = max(np.log2(p_ij / (p_i * p_j)), 0) if p_ij != 0 else None
        table[token]["PPMI"] = PPMI

    return table


if __name__ == "__main__":
    corpus = get_corpus()
    counts = get_counts(corpus)
    table = get_table(counts)
    with open("exploratory_table.pickle", "wb") as handle:
        dump(table, handle, protocol=HIGHEST_PROTOCOL)
