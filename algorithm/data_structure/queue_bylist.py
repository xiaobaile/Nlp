"""
队列：
    队列是一种数据结构，而且是一种线性的数据结构。
    特点：
        1.只允许在前端进行删除操作
        2.在后端进行插入操作
        这个模式称为FIFO先进先出（First In First Out）
    操作：
        1.进行插入操作的端称为队尾
        2.进行删除操作的端称为队头
    实现：
        栈结构通常可用数组或链表实现。。

"""

"""
队列数据结构包含的功能有：
队列是否为空 is_empty()
往队尾添加一个元素 enqueue(elem)
将队头的元素删除，并返回这个元素的值 dequeue
查看队头 peek()
打印队列 print_queue()
"""


class QueueByList(object):

    def __init__(self):
        """ 分别表示队列内的参数，队列的长度以及队列头部位置。"""
        self.entries = list()
        self.length = 0
        self.front = 0

    def enqueue(self, item):
        """ 向队列添加元素。"""
        self.entries.append(item)
        self.length += 1

    def dequeue(self):
        """ 从队列删除元素，并且更新队列。"""
        self.length -= 1
        dequeued = self.entries[self.front]
        self.front += 1
        self.entries = self.entries[self.front:]
        return dequeued

    def peek(self):
        """ 返回队列队首的元素。"""
        return self.entries[0]

    def print(self):
        print(self.entries)


if __name__ == '__main__':
    queue = QueueByList()
    queue.enqueue(12)
    queue.enqueue(34)
    queue.enqueue("abd")
    queue.enqueue(999)
    queue.print()
    print(queue.dequeue())
    queue.print()
