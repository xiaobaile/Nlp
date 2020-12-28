import os
import sys

""" 
__file__表示显示文件当前的位置
    但是：
        如果当前文件包含在sys.path里面，那么，__file__返回一个相对路径！
        如果当前文件不包含在sys.path里面，那么__file__返回一个绝对路径！
"""

print(sys.path)
# /Users/shiluyou/Desktop/Nlp/python_basic/module_os_note/os_path_basename.py
print(__file__)
# 返回文件名。os_path_basename.py
print(os.path.basename(__file__))
# 是否存在路径，不存在返回False，存在返回True。
print(os.path.exists(os.path.dirname(__file__)))
print(os.path.exists("/Users/shiluyou/Desktop/felicia"))

# /Users/shiluyou/Desktop/Nlp/python_basic/module_os_note
a = os.getcwd()
print(a)
# 返回文件路径。
b = os.path.dirname(__file__)
print(b)
# 返回文件的绝对路径。
c = os.path.abspath(b)
print(c)
