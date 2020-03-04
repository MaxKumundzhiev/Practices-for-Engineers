#Short Version of Solution
n, m = map(int,input().split())
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))

def yolo(N):
    M = 3*N
    index = 0
    dot = '.'
    line = '|'
    while index <= M:
        #loop controls number of rows
        #7 --> len of WELCOME
        first_index_map, last_index_map, middle_index_map = M // 2 - 1, M // 2 + 1, M // 2
        if (index == first_index_map) or (index == last_index_map):
            print(dot, end='')
            index += 1
        elif index == middle_index_map:
            print(line, end='')
            index += 1
        else:
            print('-', end='')
            index += 1

yolo(5)            

