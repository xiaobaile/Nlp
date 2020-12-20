import json


""" all the function in the module is used to handle file, mainly for using to write and read.
    afterwards i will add some other functions into it when we meet them.
"""


def read_txt_file(file_path: str) -> list:
    """
    read file content from .txt file and get the whole text in a list.
    :param file_path:
    :return:
    """
    with open(file_path, mode="r", encoding="utf-8") as fr:
        content = fr.readlines()
    return content


def write_txt_file(file_path: str, content: list):
    """
    write content to .txt file, resulting get file.txt.
    :param file_path:
    :param content:
    :return:
    """
    with open(file_path, mode="w", encoding="utf-8") as fw:
        for line in content:
            fw.write(str(line) + "\n")


def read_json_file(file_path: str):
    """
    read data from json file.
    :param file_path:
    :return:
    """
    with open(file_path, mode="r", encoding="utf-8") as f_json:
        content = json.load(f_json)
    return content


def write_json_file(file_path: str, content: list):
    """
    write string data into json file.
    :param file_path:
    :param content:
    :return:
    """
    with open(file_path, mode="w", encoding="utf-8") as f_json:
        json.dump(content, f_json, ensure_ascii=False)
