import time
from utils.running_time import cal_time

start_time = time.time()
"""已知三个非负整数a，b，c，满足三个数的和为target，并且a**2 + b**2 = c**2，求出所有的可能解。"""


@cal_time
def first_method(target: int):
    for a in range(0, target+1):
        for b in range(0, target+1):
            for c in range(0, target+1):
                if a + b + c == target and a ** 2 + b ** 2 == c ** 2:
                    print("a, b, c:%d, %d, %d" % (a, b, c))


@cal_time
def second_method(target: int):
    for a in range(0, target+1):
        for b in range(0, target+1):
            c = target - a - b
            if a ** 2 + b ** 2 == c ** 2:
                print("a, b, c:%d, %d, %d" % (a, b, c))


if __name__ == '__main__':
    test_times = 1000
    first_result = first_method(test_times)
    second_method(test_times)
