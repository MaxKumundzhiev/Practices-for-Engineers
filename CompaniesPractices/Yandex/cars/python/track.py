import numpy as np

from utils import Vector
from utils import convexHull, normalVector, dist, pointInsidePolygon

class Track:

    def __init__(self, innerPoints=[], outerPoints=[]):
        self.innerPoints = innerPoints
        self.outerPoints = outerPoints

        if len(self.innerPoints) == 0 and len(self.outerPoints) == 0:
            self.make_track()

        self.segments = []

        dists = [
                    dist(self.innerPoints[0].x,
                         self.innerPoints[0].y,
                         self.outerPoints[-1].x,
                         self.outerPoints[-1].y),
                    dist(self.innerPoints[0].x,
                         self.innerPoints[0].y,
                         self.outerPoints[0].x,
                         self.outerPoints[0].y),
                    dist(self.innerPoints[0].x,
                         self.innerPoints[0].y,
                         self.outerPoints[1].x,
                         self.outerPoints[1].y),
                ]

        offset = 0
        if dists[0] > dists[2]:
            offset = 1

        for i in range(len(self.innerPoints)):
            segment = [
                        self.innerPoints[i],
                        self.outerPoints[(i * 2 + len(self.outerPoints) - 1 + offset) % len(self.outerPoints)],
                        self.outerPoints[(i * 2 + len(self.outerPoints) - 1 + offset + 1) % len(self.outerPoints)]
                      ]
            self.segments.append(segment)
            segment = [
                        self.outerPoints[(i * 2 + len(self.outerPoints) - 1 + offset + 1) % len(self.outerPoints)],
                        self.innerPoints[i],
                        self.innerPoints[(i + 1) % len(self.innerPoints)],
                        self.outerPoints[((i + 1) * 2 + len(self.outerPoints) - 1 + offset) % len(self.outerPoints)]
                      ]
            self.segments.append(segment)


    def make_track(self):
        for _ in range(20000):
            x = np.random.normal() * 100
            y = np.random.normal() * 80
            self.innerPoints.append(Vector(x, y))

        self.innerPoints = convexHull(self.innerPoints)
        self.outerPoints = []
        for i in range(len(self.innerPoints)):
            a = self.innerPoints[i]
            b = self.innerPoints[(i + 1) % len(self.innerPoints)]

            n = normalVector(a, b)
            self.outerPoints.append(a + n * 50)
            self.outerPoints.append(a - n * 50)
            self.outerPoints.append(b + n * 50)
            self.outerPoints.append(b - n * 50)

        self.outerPoints = convexHull(self.outerPoints)

    def save_to(self, filename):
        with open(filename, 'w') as f:
            f.write('{}\n'.format(len(self.innerPoints)))
            for point in self.innerPoints:
                point.serialize(f)
            f.write('{}\n'.format(len(self.outerPoints)))
            for point in self.outerPoints:
                point.serialize(f)

    def withinTrack(self, point):
        return pointInsidePolygon(point, self.outerPoints) and not pointInsidePolygon(point, self.innerPoints)

    def getStartPosition(self):
        return self.innerPoints[0] - Vector(20, 0)

    @staticmethod
    def read_from(filename):
        innerPoints = []
        outerPoints = []
        with open(filename, 'r') as f:
            innerPointsLen = int(f.readline())
            for _ in range(innerPointsLen):
                point = Vector.deserialize(f)
                innerPoints.append(point)
            outerPointsLen = int(f.readline())
            for _ in range(outerPointsLen):
                point = Vector.deserialize(f)
                outerPoints.append(point)

        return Track(innerPoints, outerPoints)

    def draw(self, canvas):
        for segment in self.segments:
            for i in range(len(segment)):
                canvas.line(segment[i], segment[(i + 1) % len(segment)])
