from collections import Counter, defaultdict
from pickle import HIGHEST_PROTOCOL, dump
from typing import List

import numpy as np
import os
from convokit import Corpus, Utterance


def get_corpus() -> Corpus:
    corpus = Corpus(
        filename=os.path.join(
            os.path.dirname(__file__), os.pardir, "supreme_full_processed_lem"
        )
    )

    # Filter out speakers without gender signal
    corpus = corpus.filter_utterances_by(
        lambda utt: utt.get_speaker().retrieve_meta("gender_signal") != None
    )

    return corpus


def process_corpus(corpus: Corpus) -> List[dict]:
    """
    Each Utterance in corpus becomes a dict containing the metadata important to us.
    We do this because ConvoKit Utterance objects are not serializable using pickle.
    """
    return [
        {
            "tokens": utt.retrieve_meta("lem-tokens"),
            "gender": utt.get_speaker().retrieve_meta("gender_signal"),
        }
        for utt in corpus.iter_utterances()
    ]


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
            counts[token][utt["gender"]] += 1

    return counts


def get_table(
    counts: defaultdict(Counter), smoothing_factor: float = 1e-10
) -> defaultdict(defaultdict):
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
    # Outer layer defaultdict returns an inner defaultdict when key not present.
    # This inner defaultdict returns None when a key to it is not present.
    table = defaultdict(defaultdict)

    # The total number of times a token is used, summed for all tokens in V
    total_vocab_usage = 0
    # The total number of times a token is used by a female speaker, summed for all tokens in V
    female_total_vocab_usage = 0

    for token in counts:
        table[token]["M count"] = counts[token]["M"]
        table[token]["F count"] = counts[token]["F"]
        table[token]["total"] = table[token]["M count"] + table[token]["F count"]

        table[token]["M ratio"] = table[token]["M count"] / table[token]["total"]
        table[token]["F ratio"] = table[token]["F count"] / table[token]["total"]
        table[token]["F - M"] = table[token]["F ratio"] - table[token]["M ratio"]

        total_vocab_usage += table[token]["total"]
        female_total_vocab_usage += table[token]["F count"]

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
        PMI = np.log2(
            (p_ij + smoothing_factor)
            / ((p_i + smoothing_factor) * (p_j + smoothing_factor))
        )
        PPMI = max(PMI, 0)
        table[token]["PPMI"] = PPMI

    return table


if __name__ == "__main__":
    corpus = process_corpus(get_corpus())
    with open("processed_corpus_list.pickle", "wb") as handle:
        dump(corpus, handle, protocol=HIGHEST_PROTOCOL)

    counts = get_counts(corpus)
    table = get_table(counts)
    with open("exploratory_table.pickle", "wb") as handle:
        dump(table, handle, protocol=HIGHEST_PROTOCOL)
