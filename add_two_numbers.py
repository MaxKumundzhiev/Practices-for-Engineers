# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        def decode(n: ListNode):
            i = 1
            while n is not None:
                yield n.val * i
                i = 10 * i
                n = n.next

        def encode(i: int):
            temp = None
            for val in str (i):
                node = ListNode (val)
                node.next = temp
                temp = node
            return node

        i1 = sum (decode (l1))
        i2 = sum (decode (l2))

        return encode (i1 + i2)




