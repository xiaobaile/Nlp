import sys
import random
from cal_time import *

sys.setrecursionlimit(400000)


def _quick_sort(li, left, right):
    if left < right:  # 待排序的区域至少有两个元素
        mid = partition(li, left, right)
        _quick_sort(li, left, mid - 1)
        _quick_sort(li, mid + 1, right)


@cal_time
def quick_sort(li):
    _quick_sort(li, 0, len(li) - 1)


def partition(li, left, right):
    tmp = li[left]
    while left < right:
        while left < right and li[right] >= tmp:
            right -= 1
        li[left] = li[right]
        while left < right and li[left] <= tmp:
            left += 1
        li[right] = li[left]
    li[left] = tmp
    return left


# def partition2(li, left, right):
#     x = li[right]
#     i = left - 1
#     for j in range(left, right):
#         if li[j] <= x:
#             i += 1
#             li[i], li[j] = li[j], li[i]
#     li[i+1], li[right] = li[right], li[i+1]
#     return i+1

def random_partition(li, left, right):
    i = random.randint(left, right)
    li[i], li[left] = li[left], li[i]
    return partition(li, left, right)


# @cal_time
# def sys_sort(li):
#     li.sort()

# li = list(range(10000, 0, -1))
# #random.shuffle(li)
# quick_sort(li)

@cal_time
def quick_sort2(li):
    return _quick_sort2(li)


def _quick_sort2(li):
    if len(li) < 2:
        return li
    tmp = li[0]
    left = [v for v in li[1:] if v <= tmp]
    right = [v for v in li[1:] if v > tmp]
    left = _quick_sort2(left)
    right = _quick_sort2(right)
    return left + [tmp] + right


import copy

li = list(range(100000))
random.shuffle(li)
li2 = copy.copy(li)
quick_sort(li)
quick_sort2(li2)
