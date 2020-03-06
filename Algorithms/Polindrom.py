#Determine whether an integer is a palindrome. 
#An integer is a palindrome when it reads the same backward as forward.

class Solution:
    def isPalindrome(self, x: int) -> bool:
        number, buf_number  = str(x), str(x)[::-1]
        if number == buf_number:
            return True
        else:
            return False
