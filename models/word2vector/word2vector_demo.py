import logging
import multiprocessing
import os

import gensim
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

BASE_NAME = "/Users/shiluyou/Desktop/data"


def get_directory(base_name: str) -> tuple:
    """ 获取语料库路径以及读取语料库输出文本的路径。 """
    input_file = "zhwiki-latest-pages-articles.xml.bz2"
    output_file = "wiki.zh.text"
    input_directory = os.path.join(base_name, input_file)
    output_directory = os.path.join(base_name, output_file)
    return input_directory, output_directory


def get_model_directory(base_name: str) -> tuple:
    """ 创建模型训练结果的保存路径以及向量的保存路径。 """
    model_dir = "wiki.zh.text.model"
    vector_dir = "wiki.zh.text.vector"
    model_directory = os.path.join(base_name, model_dir)
    vector_directory = os.path.join(base_name, vector_dir)
    return model_directory, vector_directory


def read_articles(input_dir: str, output_dir: str):
    """ 将语料库的语料读取成文本数据。 """
    count = 0
    wiki_text = WikiCorpus(input_dir, lemmatize=False, dictionary=dict())
    fr = open(output_dir, mode="w", encoding="utf-8")
    for text in wiki_text.get_texts():
        fr.write(" ".join(text) + "\n")
        count += 1
        if count % 10000 == 0:
            logger.info("saving " + str(count) + " articles")
    fr.close()
    logger.info("finishing saving " + str(count) + " articles")


def train_model(output_dir: str, model_dir: str, vector_dir: str, size: int = 400, window: int = 5, min_count: int = 5):
    """ 训练模型。 """
    model = Word2Vec(LineSentence(output_dir), size=size, window=window, min_count=min_count,
                     workers=multiprocessing.cpu_count())
    model.save(model_dir)
    model.save_word2vec_format(vector_dir, binary=False)


def predict_result(model_dir: str, predict_word):
    """ 根据训练的模型对结果进行预测。 """
    model = gensim.models.Word2Vec.load(model_dir)
    result = model.most_similar(predict_word)
    for e in result:
        print(e[0], e[1])


def run():
    words = "男人"
    input_, output_ = get_directory(BASE_NAME)
    model_, vector_ = get_model_directory(BASE_NAME)
    read_articles(input_, output_)
    train_model(output_, model_, vector_)
    predict_result(model_, words)


if __name__ == '__main__':
    run()
