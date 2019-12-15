# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

#We can solve this task in a few ways:



def brackets(list):
    stack = []
    for elem in list:
        if elem == '(':
            stack.append(elem)
        else:
            if not stack:      #Если нет элемента в стеке
                return False
            stack.pop()
    return not stack

print(brackets('(())'))
print(brackets('))('))



