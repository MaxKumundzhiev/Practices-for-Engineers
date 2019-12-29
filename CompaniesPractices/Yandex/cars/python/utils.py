import numpy as np

class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def serialize(self, f):
        f.write('{}\t{}\n'.format(self.x, self.y))
    
    @staticmethod
    def deserialize(f):
        line = f.readline()
        x, y = list(map(float, line.strip().split()))
        return Vector(x, y)

    def __str__(self):
        return '{} {}'.format(self.x, self.y)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, r):
        return Vector(self.x * r, self.y * r)

    def __add__(self, rhs):
        return Vector(self.x + rhs.x, self.y + rhs.y)

    def __sub__(self, rhs):
        return Vector(self.x - rhs.x, self.y - rhs.y)

def convexHull(points):
    def goodOrientation(p, q, r):
        value = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        return value < 0

    hull = []
    points = sorted(points, key=lambda x: (x.x, x.y))
    p = 0
    while True:
        hull.append(points[p])
        q = (p + 1) % len(points)
        for i in range(len(points)):
            if goodOrientation(points[p], points[i], points[q]):
                q = i
        p = q
        if q == 0:
            break
    return hull

def normalVector(a, b):
    x = -(b.y - a.y) / (b.x - a.x)
    y = 1
    n = np.sqrt(x * x + y * y)
    x /= n
    y /= n
    return Vector(x, y)

def dist(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))

def pointInsidePolygon(point, points):
    n = len(points)
    inside = False
    x, y = point.x, point.y
    p1x, p1y = points[0].x, points[0].y
    for i in range(n+1):
        p2x, p2y = points[i % n].x, points[i % n].y
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def rotate(vector, angle):
    x = np.cos(angle) * vector.x - np.sin(angle) * vector.y
    y = np.sin(angle) * vector.x + np.cos(angle) * vector.y
    return Vector(x, y)

def constrain(value, left, right):
    if value < left:
        return left
    if value > right:
        return right

    return value