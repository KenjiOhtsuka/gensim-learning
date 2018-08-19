from os import path
import logging


def temp_file_path(relative_path):
    return path.dirname(path.realpath(__file__)) + '/tmp/' + relative_path


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm')  # comes from the first tutorial, "From strings to vectors"
print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it

index.save(temp_file_path('deerwester.index'))
index = similarities.MatrixSimilarity.load(temp_file_path('deerwester.index'))

sims = index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)  # print sorted (document number, similarity score) 2-tuples
