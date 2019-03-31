
# Напишите на Python две функции для поиска минимального значения в списке.
# Первая из них должна иметь сложность O(n2) и сравнивать каждое число со всеми другими значениями в списке;
# Вторая функция должна быть линейной с O(n) и сравнивать каждое число со всеми другими значениями в списке;



def n2(myList):
    min = myList[0]
    for i in myList:
        for j in myList:
            if j < i:
                min = j
    return min




def n(myList):
   min_value = myList[0]
   for i in myList:
       if i < min_value:
           min_value = i
   return min_value


list = [1,22,0,12]
print(n(list))
print(n2(list))