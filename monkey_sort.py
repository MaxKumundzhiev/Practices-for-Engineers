import string
import random

#str_result = 'methinks it is like a weasel'
from pip._vendor.msgpack.fallback import xrange

str_sample = '' #create a randoming string from alphabet
alphabet = string.ascii_lowercase + ' ' #create a alphabet + tab

# Generate an alphabet by myself, without manually writing

def generate_alphabet():
    i = 0
    b = 97
    length = 26
    result_alphabet = ''
    for i in range(length):
        result_alphabet += chr(b)
        b += 1
    return result_alphabet + ' '

print(generate_alphabet())
# Incrementing variable



# Conacatenation effeciancy in Python with += operator


# For Loop without range function

def main(str_result):
    #i = 0
    while monkey(len(str_result)) != str_result:
        pass
        #print(i)
        #i += 1
    print('Job is done!')



def monkey(lenght):
    result = ''
    for i in range(lenght):
        result += alphabet[random.randint(0, len(alphabet) - 1)]
    return result


#main('abbcw')


# wordlist = ['cat','dog','rabbit']
# letterlist = [ ]
# for a_word in wordlist:
#     for a_letter in a_word:
#         letterlist.append(a_letter)
# print((letterlist))



# list_1 = ['apple', 'banana']
# list_2 = [None] * 11
# i = 0
#
# for word_index in range(len(list_1)):
#     for letter_index in range(len(list_1[word_index])):
#         list_2[i] = list_1[word_index][letter_index]
#         i += 1
# print(list_2)
