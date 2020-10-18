# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

path = [0, 0, 0, 1, 0, 0]
path_len = len(path)

current_index = 0
jumps = 0

while path_len > 1:
    if path_len == 2:
        current_index += 1
        jumps += 1
        path_len -= 1
        break
    if path[current_index + 2] != 1:
        current_index += 2
        jumps += 1
        path_len -= 2
    elif path[current_index + 1] != 1:
        current_index += 1
        jumps += 1
        path_len -= 1

print(jumps)