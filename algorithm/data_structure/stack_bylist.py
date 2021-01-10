"""
栈
    栈是一种数据结构，而且是一种线性的数据结构。
    特点：
        1.只能从一端添加和删除元素（这一端称为栈顶）
        2.另一端是限制操作的（这一端称为栈底）
        这个模式称为LIFO后进先出（Last In First Out）
    操作：
        1.存储数据，称为入栈或压栈（push）
        2.提取数据，称为出栈或弹栈（pop）
    用途：
        记录浏览器的回退记录
        软件的操作撤销功能
        ide中的括号匹配检测
        代码中函数的调用和返回
        深度优先搜索
    实现：
        栈结构通常可用数组或链表实现。
"""


""" 
创建一个 Stack 类，具备以下功能：

Stack() - 创建新栈，不需要参数，返回空栈。
push(item) - 将元素添加到栈顶，需要参数，无返回值。
pop() - 删除栈顶元素，不需要参数，返回栈顶元素，并修改栈的内容。
peek() - 返回栈顶元素，不需要参数，不修改栈的内容。
isEmpty() - 检查栈是否为空，不需要参数，返回布尔值
size() - 返回栈中元素个数，不需要参数，返回整数

"""


class StackByList:
    """ 定义一个栈类，使之满足上面提到的功能。"""
    def __init__(self):
        self.stack_by = []
        self.size_by = 0

    def s_push(self, item):
        """ 压栈操作。"""
        self.stack_by.append(item)
        self.size_by += 1

    def s_pop(self):
        """ 出栈操作。"""
        pop = self.stack_by.pop()
        self.size_by -= 1
        return pop

    def s_peek(self):
        """ 查看栈顶元素。"""
        return self.stack_by[-1]

    def s_is_empty(self):
        """ 判断栈是否为空。"""
        return self.stack_by == []

    def s_size(self):
        """ 查看栈的大小。"""
        return self.size_by


if __name__ == '__main__':
    s = StackByList()
    print(s.s_is_empty())
    s.s_push(4)
    print(s)
    s.s_push('dog')
    print(s.s_peek())
    print(s.s_pop())
    print(s.s_is_empty())
