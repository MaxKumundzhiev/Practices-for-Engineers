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
    '''Simple model of Dog'''

    def __init__(self, name, age):
        '''define initialisation method of our class and name and age'''
        self.name = name
        self.age = age
        print('Dog {} is created'.format(name))

    def command_sit(self):
        '''define a method which tell the dog to sit'''
        print(self.name + ' dog sits')

    def command_jump(self):
        '''define a method which tell the  dog to jump '''
        print(self.name + ' dog jumps')

# my_dog = Dog('Dora', 10)
# neighbour_dog = Dog('Charli', 4)

# print(my_dog.name, my_dog.age, my_dog.command_sit(), my_dog.command_jump())
# print(neighbour_dog.name, neighbour_dog.age, neighbour_dog.command_sit())


class Cat():
    '''Simple model of Cat'''
    def __call__(self, name, color):
        self.name = name
        self.color = color
        print('Called the object Cat with name {0} and color {1}'.format(name, color))

# my_cat = Cat()
# my_cat('Tom', 2)



class Counter():
    '''This class counts some staff'''
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