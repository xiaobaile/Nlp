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


def select_sort(array):
    """ 选择排序。"""
    n = len(array)
    for j in range(n-1):
        min_index = j
        for i in range(j+1, n):
            if array[min_index] > array[i]:
                min_index = i
        array[min_index], array[j] = array[j], array[min_index]
        print(array)


if __name__ == '__main__':
    test_in = [i for i in range(10)]
    random.shuffle(test_in)
    print(test_in)
    select_sort(test_in)
    print(test_in)
