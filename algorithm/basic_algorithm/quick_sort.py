import random


def quick_sort(arr: list, low: int, high: int):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)


def partition(arr, low, high):
    """

    :param arr:
    :param low:
    :param high:
    :return:
    """
    i = low - 1
    pivot = arr[high]
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i+1


if __name__ == '__main__':
    test_in = [i for i in range(10)]
    n = len(test_in)
    random.shuffle(test_in)
    print(test_in)
    quick_sort(test_in, 0, n-1)
    print(test_in)
