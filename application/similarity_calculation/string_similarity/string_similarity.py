import difflib


def test():
    # 字符串相似度指的是比较两个文本相同字符个数，从而得出其相似度。
    first_sentence = "她是女生"
    second_sentence = "她是女生"
    result = difflib.SequenceMatcher(None, first_sentence, second_sentence).ratio()
    print(result)


if __name__ == '__main__':
    test()
