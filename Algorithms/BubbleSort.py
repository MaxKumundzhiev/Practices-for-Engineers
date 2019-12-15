# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------



# The main idea of bubble sort is to sort the list/array in ncreasing order
# Firstly you compare first two elements and then going till the end and loop
# Be careful, because you need to have correct amount of iterations - (it will be len(myList) - 1).
# Deeply - you need have correct last item till which we will have to compare


# def bubble_sort(myList):
#     last_item = len(myList) - 1
#     for i in range(0, last_item):
#         for j in range(0, last_item):
#             if myList[j] > myList[j+1]:
#                 myList[j], myList[j+1] = myList[j+1], myList[j]
#     return myList



## Have a look on how many times the loop executed

# def bubble_sort(myList):
#     last_item = len(myList) - 1
#     for i in range(0, last_item):
#         for j in range(0, last_item):
#             print(myList)
#             if myList[j] > myList[j+1]:
#                 myList[j], myList[j+1] = myList[j+1], myList[j]
#     return myList
#
#

## Optimized bubble sort №1

def bubble_sort_1(myList):
    count = 0
    last_item = len(myList) - 1
    for i in range(0, last_item):
        for j in range(0, last_item - i):
            if myList[j] > myList[j+1]:
                myList[j], myList[j+1] = myList[j+1], myList[j]
            count += 1
    return (myList, count)



## Optimized bubble sort №2

def bubble_sort_2(myList):
    count = 0
    last_item = len(myList) - 1
    for i in range(0, last_item):
        swapped = False
        for j in range(0, last_item - i):
            if myList[j] > myList[j+1]:
                myList[j], myList[j+1] = myList[j+1], myList[j]
                swapped = True
            count += 1
        if not swapped:
            break
    return (myList, count)






# print(list_for_change)

list_for_change = [10,75, 43, 25, -4, 27]

print(bubble_sort_1([1, 2, 3, 4, 5]))

print(bubble_sort_2([1, 2, 3, 4, 5]))




