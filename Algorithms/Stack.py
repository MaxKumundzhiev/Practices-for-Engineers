'''
Stack - data type, which can be concerned as usual glass,
in which you can OR put the element on the top OR take from the top.

Used definition "LIFO" - last in first out

Key methods:
	push() - put the element on the top of stack
	top() - get the element which is on the top
	pop() - remove the element from the top
	empty() - check whether the stack is empty or not
'''

#Sample task where stack can be used is: 'Brackets Sequence'
#Add solution to Algorithms/AmazonSoftwareIssue

#######Task
'''
On input we have breaket sequence. Check whether for each open bracket there is a close breaket. 
Start with one type of bracket: "(" and  ")"

The key idea is: 
while there is element in sequence: 
	push element on the top of stack if element is opening (or not closing) else pop "previous" element 

Border conditions:
	if we read the element but stack is empty <-- )( 
'''

