#You are asked to ensure that the first and last names of people begin with a capital letter in their passports. For example, alison heck should be capitalised correctly as Alison Heck.
#Given a full name, your task is to capitalize the name appropriately.

def capitalize(s):
    string = s.split(' ')
    if len(string) > 1:
        result = (' '.join((word.capitalize() for word in string)))
        return result 
    else: 
	return string[0]


if name == '__main__':
    s = input('Enter first name and second name')
    result = capitalize(s)
    print(result)	   

