import random


"""
二分查找就是将查找的键和子数组的中间键作比较，如果被查找的键小于中间键，就在左子数组继续查找；
如果大于中间键，就在右子数组中查找，否则中间键就是要找的元素。

注意：代码中的判断条件必须是while (left <= right)，否则的话判断条件不完整，
比如：array[3] = {1, 3, 5};待查找的键为5，此时在(low < high)条件下就会找不到，因为low和high相等时，指向元素5，但是此时条件不成立，没有进入while()中。

二分查找的变种：
    二分查找的变种和二分查找的原理一样，主要是交换判断条件（也就是边界条件）。
    例如：数组之中的数据可能重复，要求返回匹配的数据的最小（或最大）下标；
    需要找出数组中第一个大于key的元素（也就是最小的大于key的元素）的下标等等。

https://www.cnblogs.com/luoxn28/p/5767571.html
"""


def binary_search(array: list, item: int) -> bool:
    n = len(array)
    head = 0
    tail = n-1
    while head <= tail:
        middle = (head + tail) // 2
        if array[middle] == item:
            return True
        elif array[middle] > item:
            tail = middle - 1
        else:
            head = middle + 1
    return False


def binary_find(array, item):
    n = len(array)
    head = 0
    tail = n - 1
    middle = (head + tail) // 2
    while n > 0:
        if array[middle] == item:
            return True
        elif array[middle] < item:
            return binary_find(array[middle+1:], item)
        else:
            return binary_find(array[: middle], item)
    return False


def find_first_equal(array, key):
    """ 查找第一个与key相等的元素的索引。"""
    n = len(array)
    head = 0
    tail = n - 1
    while head <= tail:
        middle = (head + tail) // 2
        if array[middle] < key:
            head = middle + 1
        else:
            tail = middle - 1
        if head < n and array[head] == key:
            return head
    return False


def find_last_equal(array, key):
    """ 查找最后一个与key相等的元素的索引。"""
    n = len(array)
    head = 0
    tail = n - 1
    while head <= tail:
        middle = (head + tail) // 2
        if array[middle] > key:
            tail = middle - 1
        else:
            head = middle + 1
        if tail > 0 and array[tail] == key:
            return tail
    return False


if __name__ == '__main__':
    array_in = [i for i in range(20)]
    array_in = random.sample(array_in, 10)
    array_in = sorted(array_in)
    print(array_in)
    result = binary_find(array_in, 5)
    print(result)
    arr = [1, 2, 3, 3, 3, 3, 4, 7]
    print(find_first_equal(arr, 3))
    print(find_last_equal(arr, 3))
