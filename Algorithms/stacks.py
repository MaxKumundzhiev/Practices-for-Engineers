# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


"""Stacks - abstract data structure, similar as cup,
 where you push objects on top of each other. (LIFO paradigm - last in first out).

 Stacks operate with following operations:
 push(key) - add key on top of stack
 pop() - read and remove last object from stack
 top() - read last object from stack
 empty() - weather stack is empty

 Popular task: brackets sequence.
"""


def is_correct(sequence):
    """Check weather input sequence is correct.

    sequence: sequence of input brackets
    """

    stack = []
    for bracket in sequence:
        if bracket == '(' or bracket == '[' or bracket == '{':
            stack.append(bracket)
        else:
            if len(stack) == 0:
                return False
            top = stack.pop()
            if top == '(' and bracket != ')' or top == '[' and bracket != ']' or top == '{' and bracket != '}':
                return False

    return True if len(stack) == 0 else False


print(is_correct('([](){([])})'))


# Define own stack class
class Stack:
    """
    Stack data structure will be defined within list data structure.
    """
    def __init__(self):
        self.stack = []

    def pop(self):
        if len(self.stack) < 1:
            return
        else:
            return self.stack.pop()

    def push(self, item):
        self.stack.append(item)

    def top(self):
        if len(self.stack) < 1:
            return
        else:
            return self.stack[-1]

    def empty(self):
        return True if len(self.stack) == 0 else False


# Solve brackets sequence with defined above Class
def stack_is_correct(input_data):
    stack = Stack()

    for bracket in input_data:
        if bracket == '(' or bracket == '[':
            stack.push(bracket)

        else:
            if stack.empty():
                return False

            top = stack.pop()
            if top == '(' and bracket != ')' or top == '[' and bracket != ']':
                return False
    return stack.empty()


data = '([])'
print(stack_is_correct(data))













