from typing import List

import regex as re
from convokit import Corpus, Speaker
from convokit.text_processing import TextCleaner
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class AdvocatesProcessor:
    """Tokenizes, lemmatizes and adds gender signal metadata to supreme court corpus."""

    def __init__(self, corpus: Corpus) -> None:
        # Filter to just advocates
        self.corpus = corpus

    def process(self):
        is_advocate = lambda speaker: speaker.meta["type"] == "A"

        # Add gender signal metadata
        for speaker in self.corpus.iter_speakers():
            if is_advocate(speaker):
                gender_signal = self._get_gender_signal(speaker)
                speaker.add_meta("gender_signal", gender_signal)

        # Filter out non-advocates
        self.corpus = self.corpus.filter_utterances_by(
            lambda utt: is_advocate(utt.get_speaker())
        )

        # Filter out advocates without gender signal
        self.corpus = self.corpus.filter_utterances_by(
            lambda utt: utt.get_speaker().meta["gender_signal"] != None
        )

        # There's no need to fix what's not broken (but we can move on from this in future)
        cleaner = TextCleaner(verbosity=1000)
        print("Cleaning text")
        self.corpus = cleaner.transform(self.corpus)

        print("Lemmatizing and Tokenizing")
        for utt in self.corpus.iter_utterances():
            tokens = self.process_text(utt.text)
            utt.add_meta("lem-tokens", tokens)

        # Filter out utterances with less than 10 tokens
        self.corpus = self.corpus.filter_utterances_by(
            lambda utt: len(utt.retrieve_meta("lem-tokens")) >= 10
        )

    def listify_corpus(self) -> List[dict]:
        """
        Each Utterance in corpus becomes a dict containing the metadata important to us.
        We do this because ConvoKit Utterance objects are not serializable using pickle.
        """
        return [
            {
                "tokens": utt.retrieve_meta("lem-tokens"),
                "gender": utt.get_speaker().retrieve_meta("gender_signal"),
            }
            for utt in self.corpus.iter_utterances()
        ]

    def process_text(self, text: str) -> List[str]:
        """Returns *lemmatized* tokens from text"""
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token, pos="n") for token in word_tokenize(text)]

    def _get_gender_signal(self, advocate: Speaker) -> str:
        # Get name, split into first, middle(?) and last
        advocate_name = advocate.meta["name"].split()

        # Get the first utterance from speaker
        first_utt = self.corpus.get_utterance(advocate.get_utterance_ids()[0])
        # As found in Cai et al., advocates are introduced in utterance preceding
        # their own first utterance. Splitting by whitespace acceptable for us
        intro = self.corpus.get_utterance(first_utt.reply_to).text

        # Positive lookahead to see if there is match for advocate's last name,
        # then we discard using capture group. End result the prefix the Chief
        # Justice uses to refer to an advocate (e.g. "Mr. Shaffer")
        prefix = rf"[\w.]+(?= +{advocate_name[-1]}\b)"
        try:
            title = re.search(pattern=prefix, string=intro).group(0)
        # This means we haven't found a match for name. Skipping this case for now.
        except AttributeError:
            return None

        if title == "Mr.":
            return "M"
        elif title == "Ms.":
            return "F"
        # TODO: Ask Katie how to reference World Gender Name Dictionary
        return None

    def get_corpus(self):
        return self.corpus
