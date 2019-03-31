# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

# def find_digit(digit, myList):
#         if digit in myList:
#             print('Yes, we have %s in list' % digit)


#First of all we need to start_index, stop_index, digit and list

#Conditions:
#   1) If start_index changed with stop_index it means that the list dont have the digir which we find
#
#       if start_index > stop_index:
#               return ('List dont have such digit')

#   2) We need have a middle_index in order to split our list and recognize in which part we will continue find our digit
#      We will compare digit with middle_index
#
#         middle_index = (start_index + stop_index) // 2
#         if digit < myList[middle_index]:
#                return biniary_search(mylist, digit, start_inedx, middle_index - 1)
#         elif digit > myList[middle_index]:
#                 return binary_search(myList, digit, middle_index + 1)
#         elif myList[middle_index] == digit:
#                 return middle_index


def binary_search(myList, digit, start_index, stop_index):
    if start_index > stop_index:
        return False
    else:
        middle_index = (start_index + stop_index) // 2
        if digit == myList[middle_index]:
            return middle_index
        elif digit < myList[middle_index]:
            binary_search(myList, digit, start_index, middle_index - 1)
        else:
            binary_search(myList, digit, middle_index + 1, stop_index)


myList = [0, 2, 3, 4, 7, 10]
digit = 0
start_index = myList[0]
stop_index = len(myList)

x = binary_search(myList, digit, start_index, stop_index)

print(x)