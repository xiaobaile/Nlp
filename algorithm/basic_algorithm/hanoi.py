def hanoi(n, A, B, C):
    if n > 0:
        hanoi(n-1, A, C, B)
        print("%s->%s" % (A, C))
        hanoi(n-1, B, A, C)


hanoi(3, "A", "B", "C")
