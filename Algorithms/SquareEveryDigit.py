#For example, if we run 9119 through the function, 811181 will come out, because 92 is 81 and 12 is 1.

def square_digits(num: int) -> int:
    if num !=0 and num != 1:
        result_number = []
        buffer_number = str(num)
        [result_number.append(str(int(number)*int(number))) for number in buffer_number]
        result = int(''.join(result_number))
        return result
    else:
        return 0 if num==0 else 1
