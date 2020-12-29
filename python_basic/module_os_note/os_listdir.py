import os


"""
os.listdir(path) 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
它不包括 . 和 .. 即使它在文件夹中,只支持在 Unix, Windows 下使用。
path -- 需要列出的目录路径
返回指定路径下的文件和文件夹列表。
"""

path = "/Users/shiluyou/Desktop/Nlp/"
dirs = os.listdir(path)


for file in dirs:
    print(file)
