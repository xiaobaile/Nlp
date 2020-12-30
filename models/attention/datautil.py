import collections
import os
import re
import sys
from random import shuffle

import jieba
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.platform import gfile

jieba.load_userdict("myjiebadict.txt")

# ----------------------------------------------------------------------------------------------------------------------
# 系统字符，创建字典是需要加入
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# 文字字符替换，不属于系统字符
_NUM = "_NUM"
# ----------------------------------------------------------------------------------------------------------------------
vocab_size = 40000
data_base_dir = os.path.dirname(__file__)
data_dir = os.path.join(data_base_dir, "fanyichina")
raw_data_dir = os.path.join(data_dir, "yuliao/from")
raw_data_dir_to = os.path.join(data_dir, "yuliao/to")
vocabulary_file_en = os.path.join(data_dir, "dicten.txt")
vocabulary_file_ch = os.path.join(data_dir, "dictch.txt")
ids_dir_from = os.path.join(data_dir, "fromids")
ids_dir_to = os.path.join(data_dir, "toids")


def get_raw_file_list(path) -> tuple:
    """
    获取指定路径下的文件名和文件路径。
    :param path:
    :return: 第一个参数是文件路径，第二个参数是文件名。
    """
    files, names = list(), list()
    for f in os.listdir(path):
        if not f.endswith("~") or not f == "":
            files.append(os.path.join(path, f))
            names.append(f)
    return files, names


def get_ch_label(txt_file: str, is_ch: bool = True, normalize_digits: bool = False):
    """
    读取分词后的中文词。
    :param txt_file:
    :param is_ch:
    :param normalize_digits:
    :return:
    """
    labels = list()
    labels_sz = []
    with open(txt_file, 'rb') as f:
        for label in f:
            line_str = label.decode('utf-8')
            if normalize_digits:
                # 在line_str匹配数字用_NUM替换。
                line_str = re.sub(r'\d+', _NUM, line_str)
            no_token = basic_tokenizer(line_str)
            if is_ch:
                no_token = fen_ci(no_token)
            else:
                no_token = no_token.split()
            labels.extend(no_token)
            labels_sz.append(len(labels))
    return labels, labels_sz


def get_ch_path_text(raw_dir: str, is_ch: bool = True, normalize_digits: bool = False) -> tuple:
    """
    获取文件文本.
    :param raw_dir:
    :param is_ch:
    :param normalize_digits:
    :return:
    """
    text_files, _ = get_raw_file_list(raw_dir)
    labels = []

    training_data_szs = list([0])
    if len(text_files) == 0:
        print("err:no files in ", raw_dir)
        return labels, None
    shuffle(text_files)

    for text_file in text_files:
        training_data, training_data_sz = get_ch_label(text_file, is_ch, normalize_digits)
        training_ci = np.array(training_data)
        training_ci = np.reshape(training_ci, [-1, ])
        labels.append(training_ci)

        training_data_sz = np.array(training_data_sz) + training_data_szs[-1]
        training_data_szs.extend(list(training_data_sz))
        print("here", training_data_szs)
    return labels, training_data_szs


def basic_tokenizer(sentence: str) -> str:
    """
    对语句进行处理，通过特殊的标点符号进行切分，包括中文标点符号和英文标点符号，最终去掉字符。
    :param sentence:
    :return:
    """
    _WORD_SPLIT = "([.,!?\"':;)(])"
    _CH_WORD_SPLIT = '、|。|，|‘|’'
    str1 = ""
    for i in re.split(_CH_WORD_SPLIT, sentence):
        str1 = str1 + i
    str2 = ""
    for i in re.split(_WORD_SPLIT, str1):
        str2 = str2 + i
    return str2


def fen_ci(training_data: str) -> list:
    """
    对数据进行jie ba 分词，分词后链接再拆分。
    :param training_data:
    :return:
    """
    seg_list = jieba.cut(training_data)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    return training_ci


def create_vocabulary(vocabulary_file: str, max_vocabulary_size: int, is_ch: bool = True,
                      normalize_digits: bool = True) -> tuple:
    """
    根据输入文件创建字典。
    :param vocabulary_file:
    :param max_vocabulary_size: 40000
    :param is_ch: bool
    :param normalize_digits:
    :return:
    """
    texts, text_ssz = get_ch_path_text(raw_data_dir, is_ch, normalize_digits)
    all_words = []
    for label in texts:
        all_words += [word for word in label]

    training_label, count, dictionary, reverse_dictionary = build_dataset(all_words, max_vocabulary_size)
    if not gfile.Exists(vocabulary_file):
        if len(reverse_dictionary) > max_vocabulary_size:
            reverse_dictionary = reverse_dictionary[:max_vocabulary_size]
        with gfile.GFile(vocabulary_file, mode="w") as vocab_file:
            for w in reverse_dictionary:
                vocab_file.write(reverse_dictionary[w] + "\n")
    else:
        print("already have vocabulary!  do nothing !")
    return training_label, count, dictionary, reverse_dictionary, text_ssz


def build_dataset(words: list, n_words: int) -> tuple:
    """
    处理输入。
    :param words:
    :param n_words:
    :return:
    """
    count = [[_PAD, -1], [_GO, -1], [_EOS, -1], [_UNK, -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def create_seq2seq_file(data, source_file, target_file, text_ssz):
    """
    把data中的内存问和答ids数据  放在不同的文件里
    :param text_ssz:
    :param source_file:
    :param target_file:
    :param data:
    :return:
    """
    print("data", data, len(data))
    with open(source_file, 'w') as sor_f:
        with open(target_file, 'w') as tar_f:
            for i in range(len(text_ssz) - 1):
                print("text_ssz", i, text_ssz[i], text_ssz[i + 1], data[text_ssz[i]:text_ssz[i + 1]])
                if (i + 1) % 2:
                    sor_f.write(str(data[text_ssz[i]:text_ssz[i + 1]]).replace(',', ' ')[1:-1] + '\n')
                else:
                    tar_f.write(str(data[text_ssz[i]:text_ssz[i + 1]]).replace(',', ' ')[1:-1] + '\n')


def plot_scatter_lengths(title: str, x_title: str, y_title: str, x_lengths: list, y_lengths: list):
    """

    :param title:
    :param x_title:
    :param y_title:
    :param x_lengths:
    :param y_lengths:
    :return:
    """
    plt.scatter(x_lengths, y_lengths)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.ylim(0, max(y_lengths))
    plt.xlim(0, max(x_lengths))
    plt.show()


def plot_history_lengths(title: str, lengths: list):
    """

    :param title:
    :param lengths:
    :return:
    """
    mu = np.std(lengths)
    sigma = np.mean(lengths)
    x = np.array(lengths)
    n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.title(title)
    plt.xlabel("Length")
    plt.ylabel("Number of Sequences")
    plt.xlim(0, max(lengths))
    plt.show()


def split_file_one_line(training_data, text_ssz):
    """
    将读好的对话文本按行分开，一行问，一行答。存为两个文件。training_data为总数据，text_ssz为每行的索引
    :param text_ssz:
    :param training_data:
    :return:
    """
    source_file = os.path.join(ids_dir_from, "data_source_test.txt")
    target_file = os.path.join(ids_dir_to, "data_target_test.txt")
    create_seq2seq_file(training_data, source_file, target_file, text_ssz)


def analysis_file(source_file: str, target_file: str, plot_histograms: bool = True, plot_scatter: bool = True):
    """
    分析文本
    :param plot_scatter:
    :param plot_histograms:
    :param source_file:
    :param target_file:
    :return:
    """
    source_lengths = []
    target_lengths = []

    with gfile.GFile(source_file, mode="r") as s_file:
        with gfile.GFile(target_file, mode="r") as t_file:
            source = s_file.readline()
            target = t_file.readline()
            counter = 0

            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                num_source_ids = len(source.split())
                source_lengths.append(num_source_ids)
                num_target_ids = len(target.split()) + 1  # plus 1 for EOS token
                target_lengths.append(num_target_ids)
                source, target = s_file.readline(), t_file.readline()
    if plot_histograms:
        plot_history_lengths("target lengths", target_lengths)
        plot_history_lengths("source_lengths", source_lengths)
    if plot_scatter:
        plot_scatter_lengths("target vs source length", "source length", "target length", source_lengths, target_lengths)


def initialize_vocabulary(vocabulary_path: str) -> tuple:
    """

    :param vocabulary_path:
    :return:
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_ids(sentence: str, vocabulary: dict, normalize_digits: bool = True, is_ch: bool = True) -> list:
    """
    将句子转成ids
    :param sentence:
    :param vocabulary:
    :param normalize_digits:
    :param is_ch:
    :return:
    """
    if normalize_digits:
        sentence = re.sub(r'\d+', _NUM, sentence)
    no_token = basic_tokenizer(sentence)
    if is_ch:
        no_token = fen_ci(no_token)
    else:
        no_token = no_token.split()
    ids_data = [vocabulary.get(w, UNK_ID) for w in no_token]
    return ids_data


def text_file_to_ids_file(data_file_name: str, target_file_name: str, vocab: dict, normalize_digits: bool = True, is_ch: bool = True):
    """
    将一个文件转成ids 不是windows下的要改编码格式 utf8
    :param data_file_name:
    :param target_file_name:
    :param vocab:
    :param normalize_digits:
    :param is_ch:
    :return:
    """
    if not gfile.Exists(target_file_name):
        print("Tokenizing data in %s" % data_file_name)
        with gfile.GFile(data_file_name, mode="rb") as data_file:
            with gfile.GFile(target_file_name, mode="w") as ids_file:
                for line in data_file:
                    token_ids = sentence_to_ids(line.decode('utf8'), vocab, normalize_digits, is_ch)
                    ids_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def text_dir_to_ids_dir(text_dir: str, ids_dir: str, vocab: dict, normalize_digits: bool = True, is_ch: bool = True):
    """
    将文件批量转成ids文件
    :param ids_dir:
    :param text_dir:
    :param vocab:
    :param normalize_digits:
    :param is_ch:
    :return:
    """
    text_files, filenames = get_raw_file_list(text_dir)
    if len(text_files) == 0:
        raise ValueError("err:no files in ", text_dir)

    for text_file, name in zip(text_files, filenames):
        text_file_to_ids_file(text_file, ids_dir + name, vocab, normalize_digits, is_ch)


def ids2texts(indices, rev_vocab) -> list:
    """

    :param indices:
    :param rev_vocab:
    :return:
    """
    texts = []
    for index in indices:
        texts.append(rev_vocab[index])
    return texts


def main():
    training_data_en, count_en, dictionary_en, reverse_dictionary_en, text_ssz_en = create_vocabulary(
        vocabulary_file_en,
        vocab_size, is_ch=False,
        normalize_digits=True)

    training_data_ch, count_ch, dictionary_ch, reverse_dictionary_ch, text_ssz_ch = create_vocabulary(
        vocabulary_file_ch,
        vocab_size, is_ch=True,
        normalize_digits=True)

    vocab_en, rev_vocab_en = initialize_vocabulary(vocabulary_file_en)
    vocab_ch, rev_vocab_ch = initialize_vocabulary(vocabulary_file_ch)

    text_dir_to_ids_dir(raw_data_dir, ids_dir_from, vocab_en, normalize_digits=True, is_ch=False)
    text_dir_to_ids_dir(raw_data_dir_to, ids_dir_to, vocab_ch, normalize_digits=True, is_ch=True)

    files_from, _ = get_raw_file_list(ids_dir_from)
    files_to, _ = get_raw_file_list(ids_dir_to)
    source_train_file_path = files_from[0]
    target_train_file_path = files_to[0]
    analysis_file(source_train_file_path, target_train_file_path)


if __name__ == "__main__":
    main()
