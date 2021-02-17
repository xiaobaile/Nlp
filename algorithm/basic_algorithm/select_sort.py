import random


"""
算法思路：
    首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，
    再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
    以此类推，直到所有元素均排序完毕。
选择排序优缺点：
    优点：一轮比较只需要换一次位置；
    缺点：效率慢，不稳定。
选择排序vs冒泡排序：
（1）冒泡排序是比较相邻位置的两个数，而选择排序是按顺序比较，找最大值或者最小值；
（2）冒泡排序每一轮比较后，位置不对都需要换位置，选择排序每一轮比较都只需要换一次位置；
（3）冒泡排序是通过数去找位置，选择排序是给定位置去找数；
"""

import random
from cal_time import *


def get_min_pos(li):
    min_pos = 0
    for i in range(1, len(li)):
        if li[i] < li[min_pos]:
            min_pos = i
    return min_pos


@cal_time
def select_sort(li):
    for i in range(len(li)-1):  # n或者n-1趟
        # 第i趟无序区范围 i~最后
        min_pos = i  # min_pos更新为无序区最小值位置
        for j in range(i+1, len(li)):
            if li[j] < li[min_pos]:
                min_pos = j
        li[i], li[min_pos] = li[min_pos], li[i]


li = list(range(10000))
random.shuffle(li)
select_sort(li)


