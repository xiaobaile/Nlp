from itertools import chain


"""
你想在多个对象执行相同的操作，但是这些对象在不同的容器中，你希望代码在不
失可读性的情况下避免写重复的循环.
chain()可以把一组迭代对象串联起来，形成一个更大的迭代器：
"""
a = ["a", "b", "c", "d"]
b = [1, 2, 3, 4]
for i in chain(a, b):
    print(i)
