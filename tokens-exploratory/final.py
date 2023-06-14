from collections import Counter, defaultdict
from pickle import HIGHEST_PROTOCOL, dump
from typing import List

from convokit import Corpus, Utterance

corpus = Corpus(filename="../supreme_full_processed")

# Filter out speakers without gender signal
corpus = corpus.filter_utterances_by(
    lambda utt: utt.get_speaker().retrieve_meta("gender_signal") != None
)

# Our goal is to build a table with each word token in the vocabulary for the Supreme Court corpus,
# listing the number of male and female speakers as well as some percentages.
counts = defaultdict(Counter)
# keys: tokens in the vocabulary
# values: Counter dictionaries
#   keys: gender signals
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
# values: defaultdicts
#   keys: either "counts" or "ratio"
#   values: data for male and female speakers

for token in counts:
    table[token]["counts"] = counts[token]
    table[token]["ratio"] = defaultdict()

    table[token]["total"] = table[token]["counts"]["M"] + table[token]["counts"]["F"]

    table[token]["ratio"]["M"] = table[token]["counts"]["M"] / table[token]["total"]
    table[token]["ratio"]["F"] = table[token]["counts"]["F"] / table[token]["total"]
    table[token]["ratio"]["F - M"] = (
        table[token]["ratio"]["F"] - table[token]["ratio"]["M"]
    )


with open("exploratory_table.pickle", "wb") as handle:
    dump(table, handle, protocol=HIGHEST_PROTOCOL)
