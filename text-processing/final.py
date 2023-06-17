from convokit import Corpus, download
from ProcessingPipeline import *

# ConvoKit does not allow us to create an empty corpus that could merge all of 
# these utterances at the same time. So we do 2019 first and merge the rest in 
# a second step.
corpus = Corpus(filename=download("supreme-2019"))
processor = AdvocatesProcessor(corpus)
processor.process()
corpus = processor.get_corpus()

# Save this as a test corpus we can use later
corpus.dump("supreme_processed_lem", "..")

years = ['2016', '2017', '2018']

for year in years:
    supreme_year = Corpus(filename=download(f"supreme-{year}"))
    processor = AdvocatesProcessor(supreme_year)
    processor.process()
    corpus = corpus.merge(processor.get_corpus())

corpus.dump("supreme_full_processed_lem", "..")