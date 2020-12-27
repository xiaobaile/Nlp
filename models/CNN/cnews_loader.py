from collections import Counter

import numpy as np
import tensorflow.keras as kr


def read_file(filename: str) -> tuple:
    """ 读取文件数据.
    :param filename: 输入文件的路径。
    :return:
    """
    contents = list()
    labels = list()
    with open(filename, mode="r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines():
            label, content = line.strip().split("\t")
            if content:
                contents.append(list(content))
                labels.append(label)
    return contents, labels


def write_file(filename: str, content: list):
    """ 将内容写入文件.
    :param filename:
    :param content:
    :return:
    """
    with open(filename, mode="w", encoding="utf-8") as fw:
        fw.write("\n".join(content) + "\n")


def build_vocab(train_dir: str, vocab_dir: str, vocab_size: int = 5000):
    """ 构建词汇表，使用字符级的表示.
    :param train_dir: 读取训练数据文件，根据训练数据生成字典。
    :param vocab_dir: 生成字典的输出路径。
    :param vocab_size: 构建字典的大小，其中有一个特殊的字符"<PAD>"。
    :return:
    """
    all_data = list()
    data_train, _ = read_file(train_dir)
    for content in data_train:
        all_data.extend(content)
    """
    怎样找出一个序列中出现次数最多的元素呢？
    counter 可以将列表形式的数据转换成collections.Counter数据格式，key为列表中的元素，value为key出现的次数.
    most_common() 返回的是频率最高的词以及他们对应的次数.
    这里返回的数据结构是列表，列表中的每个元素都是元组，元组由词和词出现的次数构成.
    [('eyes', 8), ('the', 5), ('look', 4)]
    *count_pairs ---> ('eyes', 8) ('the', 5) ('look', 4)
    [('eyes', 'the', 'look'), (8, 5, 4)]
    """
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ["<PAD>"] + list(words)
    write_file(vocab_dir, words)


def read_vocab(vocab_dir: str) -> tuple:
    """ 读取词汇表，转换为{词：id}.
    :param vocab_dir:
    :return:
    """
    with open(vocab_dir, mode="r", encoding="utf-8") as fp:
        words = [line.strip() for line in fp.readlines()]
        word2id = dict(zip(words, range(len(words))))
    return words, word2id


def read_category() -> tuple:
    """ 将分类目录固定，转换为{类型：id}.
    :return:
    """
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories2id = dict(zip(categories, range(len(categories))))
    return categories, categories2id


def to_words(content: list, words: dict) -> str:
    """ 将列表转化成字符串.
    :param content:
    :param words:
    :return:
    """
    return "".join(words[x] for x in content)


def process_file(filename: str, word_to_id: dict, cat_to_id: dict, max_length: int = 600) -> tuple:
    """ 将文件转换为id表示.
    :param filename:
    :param word_to_id:
    :param cat_to_id:
    :param max_length:
    :return:
    """
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转换为one-hot表示
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    # 随机排序。
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]
