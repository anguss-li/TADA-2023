from collections import Counter, defaultdict
from pickle import HIGHEST_PROTOCOL, dump
from typing import List

import numpy as np
from convokit import Corpus, Utterance

corpus = Corpus(filename="../supreme_processed")

# Filter out speakers without gender signal
corpus = corpus.filter_utterances_by(
    lambda utt: utt.get_speaker().retrieve_meta("gender_signal") != None
)

# Our goal is to build a table with each word token in the vocabulary for the Supreme Court corpus,
# listing the number of male and female speakers as well as some percentages.
counts = defaultdict(Counter)
# keys: tokens in the vocabulary
# values: Counter dictionaries
#   keys: str, gender signals
#   values: how many speakers of given gender have said the token


def get_tokens(utt: Utterance) -> List[str]:
    """Flattens the "tokens" dictionary of an Utterance into a list."""
    return [tok["tok"] for sent in utt.retrieve_meta("tokens") for tok in sent["toks"]]


for utt in corpus.iter_utterances():
    tokens = get_tokens(utt)
    gender = utt.get_speaker().retrieve_meta("gender_signal")
    for token in tokens:
        counts[token][gender] += 1

# Add percentages in a second run because this is a bit neater.
table = defaultdict(defaultdict)
# keys: tokens in the vocabulary
# values: default dicts
#   keys: str, counts and ratios for male and female speakers
#   values: int/floats

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
    #   j: the event a female speaker speakuses a words (uses a word in V)
    PPMI = max(np.log2(p_ij / (p_i * p_j)), 0) if p_ij != 0 else 0
    table[token]["PPMI"] = PPMI

with open("exploratory_table.pickle", "wb") as handle:
    dump(table, handle, protocol=HIGHEST_PROTOCOL)
