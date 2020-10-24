# ------------------------------------------
#
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


"""
A left rotation operation on an array shifts each of the array's elements 1 unit to the left.
For example, if 2 left rotations are performed on array [1,2,3,4,5], then the array would become [3,4,5,1,2].

Given an array a of n integers and a number,d , perform d left rotations on the array.
Return the updated array to be printed as a single line of space-separated integers.
"""


a = [1, 2, 3, 4, 5]
d = 4


def rotate(a, d):
    for _ in range(d):
        if d == 0:
            return a
        else:
            item = a.pop(0)
            a.append(item)

    return a


print(rotate(a, d))