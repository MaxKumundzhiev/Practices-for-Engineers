# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


#__________________Classes___________________

#__init__ and __call__

# In [1]: class A:
#    ...:     def __init__(self):
#    ...:         print "init"
#    ...:
#    ...:     def __call__(self):
#    ...:         print "call"
#    ...:
#    ...:
#
# In [2]: a = A()
# init
#
# In [3]: a()
# call


class Dog():
    """Simple model of Dog"""

    def __init__(self, name, age):
        """define initialisation method of our class and name and age"""
        self.name = name
        self.age = age
        print('Dog {} is created'.format(name))

    def command_sit(self):
        """define a method which tell the dog to sit"""
        print(self.name + ' dog sits')

    def command_jump(self):
        """define a method which tell the  dog to jump """
        print(self.name + ' dog jumps')

# my_dog = Dog('Dora', 10)
# neighbour_dog = Dog('Charli', 4)

# print(my_dog.name, my_dog.age, my_dog.command_sit(), my_dog.command_jump())
# print(neighbour_dog.name, neighbour_dog.age, neighbour_dog.command_sit())


class Cat():
    """Simple model of Cat"""
    def __call__(self, name, color):
        self.name = name
        self.color = color
        print('Called the object Cat with name {0} and color {1}'.format(name, color))

# my_cat = Cat()
# my_cat('Tom', 2)



class Counter():
    """This class counts some staff"""
    def __init__(self, initial_count=0):
        self.count = initial_count
        print('New counter is created')

    def get(self):
        return self.count

    def increment(self):
        self.count += 1

initial_attempt = Counter()

print(initial_attempt.get())

initial_attempt.increment()
initial_attempt.increment()
initial_attempt.increment()

print(initial_attempt.get())

second_attempt = Counter(initial_count=89)
print(second_attempt.get())
second_attempt.increment()
print(second_attempt.get())



class CounterAppend():
    all_counters = [] #class attribute

    def __init__(self, initial_count=0):
        CounterAppend.all_counters.append(self)
        self.count = initial_count


counter_one = CounterAppend(92)
counter_two = CounterAppend(83)

assert len(CounterAppend.all_counters) == 2
assert counter_one.all_counters is counter_two.all_counters

print(counter_one.__class__)
print(counter_two.__class__)
print(counter_one.__dict__)
print(counter_two.__dict__)
print(counter_one.count == counter_one.__dict__['count'])


class Wierd():
    """Class this is statement"""
    f1, f2 = 0, 1
    for _ in range(10):
        f1, f2 = f2, f1 + f2

print(Wierd.f1, Wierd.f2)



class BoundMethod():
    """<bound method BoundMethod.foo of <__main__.BoundMethod object at 0x1006d6410>>
       <function BoundMethod.foo at 0x1006a84d0>"""
    def foo(self):
        pass

a = BoundMethod()
print(a.foo)
print(BoundMethod.foo)
print(a.foo is BoundMethod.foo)



class CounterProperties():
    """Properties of Class """
    def __init__(self, initial_count=0):
        self.count = initial_count

    def increment(self):
        self.count += 1

    @property
    def is_zero(self):
        return self.count == 0

c = CounterProperties()
print(c.__dict__)
assert c.is_zero #True
c.increment()
assert not c.is_zero #True



class Temperature():
    """Class for Temperature calculation"""
    def __init__(self, *, celsius=0):
        self.celsius = celsius

    @property
    def fahrenheit(self):
        return self.celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5 / 9

    @fahrenheit.deleter
    def fahrenheit(self):
        del self.celsius


c = Temperature()
c.fahrenheit = 451
assert c.celsius == 232.77777777777777
