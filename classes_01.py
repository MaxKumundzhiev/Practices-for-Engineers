# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


#__________________Classes___________________

#CLass -
#Method - def


#Creating a class (all the names of classes should start with upper case)
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

my_dog = Dog('Dora', 10)
neighbour_dog = Dog('Charli', 4)

print(my_dog.name, my_dog.age, my_dog.command_sit(), my_dog.command_jump())
print(neighbour_dog.name, neighbour_dog.age, neighbour_dog.command_sit())

