import random


"""
冒泡排序就是从数组的第一个数开始，依次和后面的数相比，若前者大则交换顺序，直到所有大的数冒到最后，最后按照从小到大排序。
冒泡排序优缺点：
    优点:比较简单，空间复杂度较低，是稳定的；
    缺点:时间复杂度太高，效率慢；
算法思路：
    1、比较相邻的元素。如果第一个比第二个大，就交换它们两个；
    2、对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对，这样在最后的元素应该会是最大的数；
    3、针对所有的元素重复以上的步骤，除了最后一个；
最好情况下O(n)，平均情况下O(n^2)，最坏情况下O(n^2)
"""


def bubble_sort(array):
    """ 冒泡排序。"""
    n = len(array)
    # j 表示第n趟，一共n趟或者n-1趟
    for j in range(n-1):
        # 如果某趟没有进行交换，表示已经按照序列排列，直接退出。
        exchange = False
        # 第i趟，无序区[0, n-i-1], i表示箭头0~n-i-2
        for i in range(n-1-j):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                exchange = True
        print("finish %d times sort..." % j)
        if not exchange:
            break


if __name__ == '__main__':
    test_in = [i for i in range(10)]
    random.shuffle(test_in)
    print(test_in)
    bubble_sort(test_in)
    print(test_in)