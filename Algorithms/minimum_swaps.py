# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


"""
You are given an unordered array consisting of consecutive integers as [1, 2, 3, ..., n] without any duplicates.
You are allowed to swap any two elements.
You need to find the minimum number of swaps required to sort the array in ascending order.
"""


array = [2, 3, 89, -2]
size = len(array)

swaps = 0

minimum_item = array[0]
minimum_index = 0

for item_index in range(1, size):
    if array[item_index] < minimum_item:
        pass

