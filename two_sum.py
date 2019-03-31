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
            if result == target:
                return result
            elif result != target:
                result = 'Fasle'
    return result


print((twoSum([2,7,11,15], 22))


