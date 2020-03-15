#pragma once

#include "utils.h"

class TTrack {
public:
    TTrack(const std::vector<Vector>& innerPoints,
          const std::vector<Vector>& outerPoints)
    : InnerPoints(innerPoints)
    , OuterPoints(outerPoints)
    {
        std::vector<double> dists = {
            Dist(InnerPoints[0].X,
                 InnerPoints[0].Y,
                 OuterPoints[OuterPoints.size() - 1].X,
                 OuterPoints[OuterPoints.size() - 1].Y),
            Dist(InnerPoints[0].X,
                 InnerPoints[0].Y,
                 OuterPoints[0].X,
                 OuterPoints[0].Y),
            Dist(InnerPoints[0].X,
                 InnerPoints[0].Y,
                 OuterPoints[1].X,
                 OuterPoints[1].Y)
        };

        int offset = 0;
        if (dists[0] > dists[2]) {
            offset = 1;
        }

        for (int i = 0; i < InnerPoints.size(); ++i) {
            {
                std::vector<Vector> segment = {
                    InnerPoints[i],
                    OuterPoints[(i * 2 + OuterPoints.size() - 1 + offset) % OuterPoints.size()],
                    OuterPoints[(i * 2 + OuterPoints.size() - 1 + offset + 1) % OuterPoints.size()]
                };
                Segments.push_back(segment);
            }
            {
                std::vector<Vector> segment = {
                    OuterPoints[(i * 2 + OuterPoints.size() - 1 + offset + 1) % OuterPoints.size()],
                    InnerPoints[i],
                    InnerPoints[(i + 1) % InnerPoints.size()],
                    OuterPoints[((i + 1) * 2 + OuterPoints.size() - 1 + offset) % OuterPoints.size()]
                };
                Segments.push_back(segment);
            }
        }
    }

    template<typename TIn>
    static TTrack ReadFrom(TIn& in) {
        std::vector<Vector> innerPoints;
        std::vector<Vector> outerPoints;
        int innerPointsLen = 0;
        int outerPointsLen = 0;

        in >> innerPointsLen;
        for (int i = 0; i < innerPointsLen; ++i) {
            Vector point = Vector::Deserialize(in);
            innerPoints.push_back(point);
        }

        in >> outerPointsLen;
        for (int i = 0; i < outerPointsLen; ++i) {
            Vector point = Vector::Deserialize(in);
            outerPoints.push_back(point);
        }

        return TTrack(innerPoints, outerPoints);
    }

    bool WithinTrack(Vector point) {
        return PointInsidePolygon(point, OuterPoints) && !PointInsidePolygon(point, InnerPoints);
    }

    Vector GetStartPosition() {
        return InnerPoints[0] - Vector(20, 0);
    }

    const std::vector<std::vector<Vector>>& GetSegments() const {
        return Segments;
    }
    const std::vector<Vector>& GetSegment(int i) const {
        return Segments[i];
    }

private:
    std::vector<Vector> InnerPoints;
    std::vector<Vector> OuterPoints;
    std::vector<std::vector<Vector>> Segments;
};
