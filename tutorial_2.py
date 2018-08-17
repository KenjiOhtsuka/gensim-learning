from os import path

def temp_file_path(relative_path):
    return path.dirname(__file__) + '/tmp/' + relative_path

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
if (path.exists(temp_file_path("deerwester.dict"))):
    dictionary = corpora.Dictionary.load(temp_file_path('deerwester.dict'))
    corpus = corpora.MmCorpus(temp_file_path('deerwester.mm'))
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

################################################################################
# To bring out hidden structure in the corpus, discover relationships between words and use them to describe the documents in a new and (hopefully) more semantic way.
# To make the document representation more compact. This both improves efficiency (new representation consumes less resources) and efficacy (marginal data trends are ignored, noise-reduction).
################################################################################

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(3)

for doc in corpus_lsi:
    print(doc)

lsi.save(temp_file_path('model.lsi'))
lsi = model.LsiModel.load(temp_file_path('model.lsi'))

# model = models.TfidfModel(corpus, normalize=True)
# model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
