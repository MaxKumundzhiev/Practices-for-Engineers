#pragma once

#include <vector>
#include <cmath>

class Vector {
public:
    Vector(double x, double y) {
        X = x;
        Y = y;
    }

    template <typename TOut>
    void Serialize(TOut& out) {
        out << X << "\t" << Y << std::endl;
    }

    template <typename TIn>
    static Vector Deserialize(TIn& in) {
        double x, y;
        in >> x >> y;
        return Vector(x, y);
    }

    Vector operator*(double r) {
        return Vector(X * r, Y * r);
    }

    Vector operator+(const Vector& rhs) {
        return Vector(X + rhs.X, Y + rhs.Y);
    }

    Vector operator-(const Vector& rhs) {
        return Vector(X - rhs.X, Y - rhs.Y);
    }

    bool operator<(const Vector& rhs) const {
        return (X < rhs.X) && (Y < rhs.Y);
    }

    double X, Y;
};

bool GoodOrientation(const Vector& p, const Vector& q, const Vector& r) {
    double value = (q.Y - p.Y) * (r.X - q.X) - (q.X - p.X) * (r.Y - q.Y);
    return value < 0;
}

std::vector<Vector> ConvexHull(std::vector<Vector> points) {
    std::vector<Vector> hull;
    std::sort(points.begin(), points.end());
    int p = 0;
    for (;;) {
        hull.push_back(points[p]);
        int q = (p + 1) % points.size();
        for (int i = 0; i < points.size(); ++i) {
            if (GoodOrientation(points[p], points[i], points[q])) {
                q = i;
            }
        }
        p = q;
        if (q == 0) {
            break;
        }
    }

    return hull;
}

Vector NormalVector(Vector a, Vector b) {
    double x = -(b.Y - a.Y) / (b.X - a.X);
    double y = 1;
    double n = sqrt(x * x + y * y);
    x /= n;
    y /= n;
    return Vector(x, y);
}

double Dist(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

bool PointInsidePolygon(Vector point, const std::vector<Vector>& points) {
    int n = points.size();
    bool inside = false;
    double x = point.X;
    double y = point.Y;
    double p1x = points[0].X;
    double p1y = points[0].Y;

    double xinters = 0;

    for (int i = 0; i < n + 1; ++i) {
        double p2x = points[i % n].X;
        double p2y = points[i % n].Y;
        if (y > std::min(p1y, p2y)) {
            if (y <= std::max(p1y, p2y)) {
                if (x <= std::max(p1x, p2x)) {
                    if (p1y != p2y) {
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x;
                    }
                    if (p1x == p2x || x <= xinters) {
                        inside = !inside;
                    }
                }
            }
        }
        p1x = p2x;
        p1y = p2y;
    }

    return inside;
}

Vector Rotate(Vector vector, double angle) {
    double x = cos(angle) * vector.X - sin(angle) * vector.Y;
    double y = sin(angle) * vector.X + cos(angle) * vector.Y;

    return Vector(x, y);
}

double Constrain(double value, double left, double right) {
    if (value < left) {
        return left;
    }
    if (value > right) {
        return right;
    }
    return value;
}
