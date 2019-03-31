list = [1,2,6,4]


def diff(list):
    min_value = list[0]
    diff_value = -1
    for i in list:
        if i <= min_value:
            min_value = i
        elif (i - min_value) > diff_value:
            diff_value = i - min_value
    return diff_value


print(diff(list))