class Rational:
    @staticmethod
    def _gcd(m, n):
        """
        定义一个求最大公约数的函数。
        建立有理数，应考虑约去其分子和分母的最大公约数，避免无意义的资源浪费。
        gcd的参数应该是两个整数，它们不属于被定义的有理数类型；
        gcd的计算并不依附任何有理数类的对象，因此其参数表中似乎不应该以表示有理数的self作为第一个参数。
        gcd是为有理数类的实现而需要使用的一种辅助功能，根据信息局部化的元组，局部使用的功能不应该定义为全局函数。
        因此，gcd应该是在有理数类里定义的一个非实例方法。
        python把在类里定义的这种方法称为静态方法，以修饰符@staticmethod标识，没有self参数。
        本质上说，静态方法就是在类里面定义的普通函数，但也是该类的局部函数。
        :param m:
        :param n:
        :return:
        """
        if n == 0:
            m, n = n, m
        while m != 0:
            m, n = n % m, m
        return n

    def __init__(self, num, den=1):
        """
        1。检查参数类型是否合适
        2。检查分母是否为0
        3。初始化方法的实参可能有正有负
        :param num:
        :param den:
        """
        if not isinstance(num, int) or not isinstance(den, int):
            raise TypeError
        if den == 0:
            raise ZeroDivisionError
        sign = 1
        if num < 0:
            num, sign = -num, -sign
        if den < 0:
            den, sign = -den, -sign
        # 两个属性都是内部属性，不应该在类之外去引用它们。
        self._num = sign * (num//g)
        self._den = den // g

    def num(self):
        return self._num

    def den(self):
        return self._den

    def __add__(self, other):
        den = self._den * other.den()
        num = self._num * other.den() + self._den * other.num()
        return Rational(num, den)

    def __str__(self):
        """
        为了便于输出等目的，经常在类里定义一个把该类的对象转换到字符串的方法。
        :return:
        """
        return str(self._num) + "/" + str(self._den)


class Countable:
    # 类属性，
    counter = 0

    def __init__(self):
        # 类方法调用类属性的方式。
        # 类作用域里的局部名字与函数作用域里局部名字有不同的规定。
        Countable.counter += 1

    @classmethod
    def get_count(cls):
        """
        类方法实现与本类的所有对象有关的操作。
        需要通过这种点的形式进行访问，因为在类里定义的名字，其作用域并不会自动延伸到内部嵌套作用域。
        :return:
        """
        return Countable.counter


def func(arg=None):
    # 函数作用域的局部变量。
    variables = "hello"

    def inner_func():
        # 在进行应用是不需要通过点的形式进行访问。
        # 这是因为，在函数中局部名字的作用域自动延伸到内部嵌套的作用域。
        print(variables + "world")

    inner_func()


"""
从程序的正文看，正在执行的方法f定义在类B里，在类B里，self的类型应该是。
如果根据这个类型去查找g，就应该找到类B里定义的函数g。
采用这种根据静态程序正文去确定被调用方法的规则称为静态约束（静态绑定）。
但在python里不这样做，它和多数常见的面向对象语言一样，基于方法调用self所表示的那个实例对象的类型去确定应该调用哪个g，
这种方式被称为动态约束。
"""


class B:
    def f(self):
        self.g()

    def g(self):
        print("B.G called.")


class C:
    def g(self):
        print("C.G called.")


if __name__ == "__main__":
    func()
    x = C()
    x.g()


