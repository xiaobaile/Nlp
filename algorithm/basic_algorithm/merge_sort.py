import random
from cal_time import *


def merge2list(li1, li2):
    li = []
    i = 0
    j = 0
    while i < len(li1) and j < len(li2):
        if li1[i] <= li2[j]:
            li.append(li1[i])
            i += 1
        else:
            li.append(li2[j])
            j += 1
    while i < len(li1):
        li.append(li1[i])
        i += 1
    while j < len(li2):
        li.append(li2[j])
        j += 1
    return li


def merge(li, low, mid, high):
    # 列表两段有序: [low, mid] [mid+1, high]
    i = low
    j = mid + 1
    li_tmp = []
    while i <= mid and j <= high:
        if li[i] <= li[j]:
            li_tmp.append(li[i])
            i += 1
        else:
            li_tmp.append(li[j])
            j += 1
    while i <= mid:
        li_tmp.append(li[i])
        i += 1
    while j <= high:
        li_tmp.append(li[j])
        j += 1
    # li_tmp[0:high-low+1] li[low:high+1]
    for i in range(low, high + 1):
        li[i] = li_tmp[i - low]
    # li[low:high+1] = li_tmp


def _merge_sort(li, low, high):  # 排序li的low到high的范围
    if low < high:
        mid = (low + high) // 2
        _merge_sort(li, low, mid)
        _merge_sort(li, mid + 1, high)
        # print(li[low:mid + 1], li[mid + 1:high + 1])
        merge(li, low, mid, high)
        # print(li[low: high + 1])


@cal_time
def merge_sort(li):
    _merge_sort(li, 0, len(li) - 1)


li = list(range(100000))
li.sort()
random.shuffle(li)
merge_sort(li)
# print(li)
