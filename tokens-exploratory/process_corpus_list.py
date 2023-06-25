# Get what we actually need from corpus
from collections import defaultdict
from typing import List, TypeVar

from final import get_corpus, process_corpus
from pickle import HIGHEST_PROTOCOL, dump

processed = process_corpus(get_corpus())

with open("processed_corpus_list.pickle", "wb") as handle:
    dump(processed, handle, protocol=HIGHEST_PROTOCOL)
