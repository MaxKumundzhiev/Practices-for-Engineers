# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


array_len = 2
array = [1, 3, 2]

swaps = 0

for i in range(array_len-1):
    for j in range(0, array_len-i-1):
        if array[j] > array[j+1]:
            array[j], array[j + 1] = array[j + 1], array[j]
            swaps += 1

print(array)
print(swaps)


while True:
    swapFlag = False
    for i in range(array_len-1):
        if array[i] > array[i+1]:
            array[i], array[i+1] = array[i+1], array[i]
            swaps += 1
            swapFlag = True
    if not swapFlag:
        break
