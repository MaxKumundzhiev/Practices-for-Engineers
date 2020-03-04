#  Given a string, find the length of the longest substring without repeating characters.
 
string = (input())
my_list = []
first_character = string[0]
lenght = 0
print(string)

for index in range(len(string)):
  if string[index] not in my_list:
    my_list.append(string[index])
    lenght += 1
    if string[index] != string[index + 1]:
      continue
print(my_list, lenght)  

