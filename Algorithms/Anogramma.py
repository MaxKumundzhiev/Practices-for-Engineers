

# def alphabet(lenght):
#     alpha = ''
#     buf = ord('a')
#     for i in range(lenght):
#         alpha += (chr(buf))
#         buf += 1
#     return alpha
#
# print(alphabet(26))

import random
import string

# def random_alphabet(length):
#     a = ord('a')
#     alphabet = []
#     for i in range(length):
#         alphabet.append(chr(a))
#         a += 1
#     return alphabet

#print(random_alphabet(26), type(random_alphabet(26)))

# def anogramma(list1,list2):
#     return ((list1 == list2), (list1, list2))

def anagramSolution2(s1,s2):
    alist1 = list(s1)
    alist2 = list(s2)

    alist1.sort()
    alist2.sort()

    pos = 0
    matches = True

    while pos < len(s1) and matches:
        if alist1[pos]==alist2[pos]:
            pos = pos + 1
        else:
            matches = False

    return matches

def max(list):
    sum = 0
    for i in list:
        sum += i
    return sum

list = [1,2,36]

print(max(list))






# def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
#     return ''.join(random.choice(chars) for _ in range(size))
#list1 = id_generator()
#list2 = id_generator()

