import numpy as np

from utils import Vector, pointInsidePolygon, rotate, constrain

class Car:

    def __init__(self, track):
        self.track = track
        self.position = track.getStartPosition()
        self.direction = Vector(0, -1)
        self.v = 1

        self.broken = False
        self.iteration = 0
        self.currentSegment = 0

    def update(self):
        if self.broken:
            return

        if self.iteration > 1000:
            self.broken = True
            return

        for i, segment in enumerate(self.track.segments):
            if pointInsidePolygon(self.position, segment):
                if not (i == self.currentSegment or i - self.currentSegment in [1, 2]):
                    self.broken = True
                else:
                    self.currentSegment = max(self.currentSegment, i)
                break


        self.position += self.direction * self.v
        self.iteration += 1

    def forward(self):
        self.v = constrain(self.v + 0.1, 0, 5)

    def backward(self):
        self.v = constrain(self.v / 2 - 0.01, 0, 5)

    def left(self):
        if self.v == 0:
            return
        self.direction = rotate(self.direction, 0.1 / self.v)

    def right(self):
        if self.v == 0:
            return
        self.direction = rotate(self.direction, -0.1 / self.v)

class Environment:

    def __init__(self, track):
        self.track = track
        self.car = Car(track)

    def step(self, action):
        [self.car.forward, self.car.backward, self.car.left, self.car.right][action]()
        self.car.update()

        if not self.track.withinTrack(self.car.position):
            self.car.broken = True

        return self.car.broken, len(self.track.segments) - 1 == self.car.currentSegment, float(self.car.currentSegment) / float(len(self.track.segments) - 1)

    def state(self):
        leftDistance = 0
        rightDistance = 0
        frontDistance = 0

        for d, p in [(30, 1), (15, 0.5), (7.5, 0.25)]:
            point = self.car.position + rotate(self.car.direction, np.pi/4) * d
            if self.track.withinTrack(point):
                rightDistance = p
                break

        for d, p in [(30, 1), (15, 0.5), (7.5, 0.25)]:
            point = self.car.position + rotate(self.car.direction, -np.pi/4) * d
            if self.track.withinTrack(point):
                leftDistance = p
                break

        for d, p in [(100, 1), (50, 0.5), (25, 0.25)]:
            point = self.car.position + self.car.direction * d
            if self.track.withinTrack(point):
                frontDistance = p
                break

        return [self.car.v, rightDistance, leftDistance, frontDistance]