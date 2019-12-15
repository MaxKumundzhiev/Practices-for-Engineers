# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


def twoSum(list, target):
    for i in range(len(list)):
        for j in range(1, len(list)):
            result = list[i] + list[j]
            L_i = list[i]
            L_j = list[j]
            if (result == target) and (i != j):
                return (i, j)
            elif result != target:
                result = 'Fasle'
    return result


print((twoSum([-1,-2,-3,-4,-5], -8)))


