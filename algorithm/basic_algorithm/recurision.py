from cal_time import *


# 斐波那契数列 1 1 2 3 5 8 ...

# F(n) = F(n-1) + F(n-2)   F(0)=1 F(1)=1


def fibnacci(n):  # O(2^n)
    if n == 0 or n == 1:
        return 1
    else:
        return fibnacci(n - 1) + fibnacci(n - 2)


@cal_time
def fib1(n):
    return fibnacci(n)


@cal_time
def fib2(n):
    li = [1, 1]
    for i in range(2, n + 1):
        li.append(li[-1] + li[-2])
    return li[n]


@cal_time
def fib3(n):
    a = 1
    b = 1
    c = 1
    for i in range(2, n + 1):
        c = a + b
        a = b
        b = c
    return c


fib2(10000)
