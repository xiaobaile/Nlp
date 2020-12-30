import os


""" os.pardir 本质上是..
"""
print(os.path.dirname(__file__))
print(os.path.join(os.path.dirname(__file__), os.pardir))
# /Users/shiluyou/Desktop/Nlp/python_basic/module_os_note/..

print(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
print(os.path.join(os.pardir, os.path.dirname(__file__), os.pardir, os.pardir))

base = os.getcwd()
dir_ = os.path.join(base, os.pardir)
print(dir_)