# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


def training():
    tests_count = int(input())
    for i in range(1, tests_count + 1):
        n, p = [int(s) for s in input().split(" ")]
        result = None
        skills = [int(s) for s in input().split(" ")]
        skills.sort()
        skills.reverse()
        for team in range(n - p + 1):
            ttime = 0
            for elem in skills[team+1:team+p]:
                ttime += skills[team] - elem
            if (result == None) or (result > ttime):
                result = ttime
        print('Case #%d: %d' % (i, result))




training()









