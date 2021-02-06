from numpy import *


"""
在Numpy中经常使用到的操作由扁平化操作,Numpy提供了两个函数进行此操作,他们的功能相同,但在内存上有很大的不同.
可以看到这两个函数实现的功能一样,但我们在平时使用的时候flatten()更为合适.
在使用过程中flatten()分配了新的内存,但ravel()返回的是一个数组的视图.
都是按行进行展开。
"""
a = arange(12).reshape(3, 4)
print(a)
print(a.ravel())
print(a.flatten())
