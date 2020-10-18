# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


"""Queues - abstract data structure, similar as usual daily queue in shop.
    Last element added in the end of the queue and first will be served.
    In other words, FIFO (first in, first out)

    Queues can be implemented using stacks as well.

    Queue operate with following operations:
    front - beginning of queue
    back - end of queue

    enqueue (pushback(item)) - add element to the end of queue
    dequeue (popfront(item)) - add element to the beginning of queue
    empty() - weather queue is empty
"""


class Queue:
    """
    Notes: Realization within LIST
    """

    def __init__(self):
        self.queue = []

    def pushback(self, item):
        self.queue.append(item)

    def popfront(self):
        return self.queue.pop(0)

    def empty(self):
        return True if len(self.queue) == 0 else False


my_queue = Queue()
my_queue.pushback(10)
my_queue.popfront()
print(my_queue.empty())


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


left_stack = Stack()
right_stack = Stack()

for item in range(6):
    left_stack.push(item)




