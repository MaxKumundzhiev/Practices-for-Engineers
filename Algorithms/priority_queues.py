# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


"""Queues with priorities - abstract data structure, similar as usual daily queue in shop.
    Last element added in the end of the queue and first will be served.
    The main difference between simple queues is that once element added, it has also it's own priority.

    And we process element, who's priority is maximum.

    Queue operate with following operations:
    front - beginning of queue
    back - end of queue

    insert(item, priority) - add element to the queue
    remove(priority) - remove element to the queue
    get_max() - get element with max priority
    extract_max() - retrieve element with max priority
    change_priority(item, priority) - change priority of item on provided

    Used in:
    Dextra algorithm, Prima algorithm, Hoffman algorithm, Heap sort

    Heaps - the way of realisation of queues with priority.
"""


ar = [1, 1, 3, 1, 2, 1, 3, 3, 3, 3]

# socks_dict = dict()
# print(bool(socks_dict))
# count = 1
#
# for sock in ar:
#     if not bool(socks_dict):
#         socks_dict[sock] = sock
#     else:
#         socks_dict.update({sock: count+1})
#
# print(socks_dict)

# result = 0
# unique = set(ar)
# socks_items = [ar.count(sock) for sock in unique]
# for pair in socks_items:
#     amount = pair % 2
#     if amount == 0:
#         result += pair / 2
#     else:
#         result += (pair - 1) / 2
#
# print(result)







