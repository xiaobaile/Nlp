from simhash import Simhash


def sim_hash_similarity(text1, text2):
    """
    :param text1: 文本1
    :param text2: 文本2
    :return: 返回两篇文章的相似度
    """
    aa_sim_hash = Simhash(text1)
    bb_sim_hash = Simhash(text2)
    max_hash_bit = max(len(bin(aa_sim_hash.value)), (len(bin(bb_sim_hash.value))))
    # 汉明距离
    distance = aa_sim_hash.distance(bb_sim_hash)
    similar = 1 - distance / max_hash_bit
    return similar


if __name__ == '__main__':
    print(sim_hash_similarity('在历史上有著许多数学发现', '在历史上有著许多科学发现'))