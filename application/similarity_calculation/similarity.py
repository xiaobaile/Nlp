from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.bm25 import BM25
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances


class Similarity(object):
    def __init__(self, corpus=None):
        self.corpus = corpus
        if self.corpus:
            self.tf_idf_vectorizer = self.get_tf_idf_vectorizer(self.corpus)
            self.corpus_vector = self.tf_idf_vectorizer.transform(self.corpus)
            self.bm25_model = BM25([s.split() for s in corpus])
            self.average_idf = sum(map(lambda k: float(self.bm25_model.idf[k]), self.bm25_model.idf.keys())) / len(
                self.bm25_model.idf.keys())

    @staticmethod
    def get_tf_idf_vectorizer(self, corpus):
        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf_vectorizer.fit(corpus)
        return tf_idf_vectorizer

    def get_vector(self, query):
        vector = self.tf_idf_vectorizer.transform([query])
        return vector[0]

    def similarity(self, query, type):
        assert self.corpus is not None, "corpus can not be None"
        ret = []
        if type == "cosine":
            query = self.get_vector(query)
            for item in self.corpus_vector:
                similarity = cosine_similarity(item, query)
                ret.append(similarity[0][0])
        elif type == "manhattan":
            query = self.get_vector(query)
            for item in self.corpus_vector:
                similarity = manhattan_distances(item, query)
                ret.append(similarity[0][0])
        elif type == "euclidean":
            query = self.get_vector(query)
            for item in self.corpus_vector:
                similarity = euclidean_distances(item, query)
                ret.append(similarity[0][0])
        elif type == "bm25":
            query = query.split()
            ret = self.bm25_model.get_scores(query)
        else:
            raise ValueError("similarity type error : %s" % type)
        return ret


if __name__ == "__main__":
    corpus_ = ['帮我 打开 灯', '打开 空调', '关闭 空调', '关灯', '音量 调高', '声音 调高']
    sim = Similarity(corpus_)
    print(sim.similarity('打开 灯', 'cosine'))
    print(sim.similarity('打开 灯', 'manhattan'))
    print(sim.similarity('打开 灯', 'euclidean'))
    print(sim.similarity('打开 灯', 'bm25'))
