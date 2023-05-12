class Test:
    count = 0
    __age = 2

    def __init__(self, name):
        self.count = 2
        self.name = name
        print(self.__age)


print(Test.count)
test1 = Test("张三")
print(test1.count)
print(Test.count)
print(test1._Test__age)
