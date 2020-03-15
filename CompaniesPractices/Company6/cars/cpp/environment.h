#pragma once

#include "track.h"
#include "utils.h"

class TCar {
public:
    TCar(TTrack& track)
    : Track(track)
    , Position(track.GetStartPosition())
    , Direction(0, -1)
    , V(1)
    , Broken(false)
    , Iteration(0)
    , CurrentSegment(0)
    {}

    void Update() {
        if (Broken) {
            return;
        }

        if (Iteration > 900) {
            Broken = true;
            return;
        }

        for (int i = 0; i < Track.GetSegments().size(); ++i) {
            const std::vector<Vector>& segment = Track.GetSegment(i);
            if (PointInsidePolygon(Position, segment)) {
                if (!(i == CurrentSegment || (i - CurrentSegment == 1) || (i - CurrentSegment == 2))) {
                    Broken = true;
                } else {
                    CurrentSegment = std::max(CurrentSegment, i);
                }
                break;
            }
        }

        Position = Position +  Direction * V;
        Iteration += 1;

    }

    void Forward() {
        V = Constrain(V + 0.1, 0, 5);
    }

    void Backward() {
        V = Constrain(V / 2 - 0.01, 0, 5);
    }

    void Left() {
        if (V == 0) {
            return;
        }
        Direction = Rotate(Direction, 0.1 / V);
    }

    void Right() {
        if (V == 0) {
            return;
        }
        Direction = Rotate(Direction, -0.1/V);
    }

private:
    TTrack& Track;
    Vector Position;
    Vector Direction;
    double V;

    bool Broken;
    int Iteration;
    int CurrentSegment;

    friend class TEnvironment;
};

struct TStepResult {
    bool Broken;
    bool Done;
    double Progress;
};

class TEnvironment {
public:
    TEnvironment(TTrack& track)
    : Track(track)
    , Car(Track)
    {}

    TStepResult Step(int action) {
        if (action > 3 || action < 0) {
            throw std::runtime_error("action > 3 || action < 0");
        }

        switch (action) {
            case 0:
                Car.Forward();
                break;
            case 1:
                Car.Backward();
                break;
            case 2:
                Car.Left();
                break;
            case 3:
                Car.Right();
                break;
        };

        Car.Update();

        if (!Track.WithinTrack(Car.Position)) {
            Car.Broken = true;
        }

        return {
                Car.Broken,
                Track.GetSegments().size() - 1 == Car.CurrentSegment,
                double(Car.CurrentSegment) / double(Track.GetSegments().size() - 1)
               };


    }

    std::vector<double> State() {
        double leftDistance = 0;
        double rightDistance = 0;
        double frontDistance = 0;

        std::vector<std::pair<double, double>> leftAndRight{{30, 1}, {15, 0.5}, {7.5, 0.25}};
        std::vector<std::pair<double, double>> forward{{100, 1}, {50, 0.5}, {25, 0.25}};

        for (auto dp : leftAndRight) {
            double d = dp.first;
            double p = dp.second;
            Vector point = Car.Position + Rotate(Car.Direction, M_PI/4) * d;
            if (Track.WithinTrack(point)) {
                rightDistance = p;
                break;
            }
        }

        for (auto dp : leftAndRight) {
            double d = dp.first;
            double p = dp.second;
            Vector point = Car.Position + Rotate(Car.Direction, -M_PI/4) * d;
            if (Track.WithinTrack(point)) {
                leftDistance = p;
                break;
            }
        }

        for (auto dp : forward) {
            double d = dp.first;
            double p = dp.second;
            Vector point = Car.Position + Car.Direction * d;
            if (Track.WithinTrack(point)) {
                frontDistance = p;
                break;
            }
        }

        return {Car.V, rightDistance, leftDistance, frontDistance};
    }

private:
    TTrack& Track;
    TCar Car;
};
