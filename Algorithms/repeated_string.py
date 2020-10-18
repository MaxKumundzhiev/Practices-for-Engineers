# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""
Lilah has a string, s, of lowercase English letters that she repeated infinitely many times.
Given an integer, n, find and print the number of letter a's in the first n letters of Lilah's infinite string.
For example, if the string s='abcac' and n=10, the substring we consider is 'abcacabcac',
the first 10 characters of her infinite string. There are 4 occurrences of a in the substring.
"""

s = 'aba'
n = 10

string = (s * (n+1))
string[:n].count('a')


def RepeatedString(s, n):
    return s.count('a') * (n // s) + s[:n % len(n)]


def repeatedString(s, n):
    initial_string = s
    final_string_len = n
    initial_string_len = len(s)

    if 'a' in set(initial_string) and len(set(initial_string)) == 1:
        return int(final_string_len)

    if initial_string_len > final_string_len:
        return initial_string[:final_string_len]

    elif initial_string_len == final_string_len:
        return initial_string

    else:
        difference = final_string_len - initial_string_len
        whole_word_entrances = difference // initial_string_len
        if (whole_word_entrances * initial_string_len) + initial_string_len !=  final_string_len:
            partial_word_entrances = final_string_len - (initial_string_len + initial_string_len * whole_word_entrances)
            result = initial_string + initial_string * whole_word_entrances + initial_string[:partial_word_entrances+1]
        else:
            result = initial_string + initial_string * whole_word_entrances
    return result

# print(repeatedString(s, n))