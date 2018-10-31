#include <boost/math/quadrature/gauss.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "utils.h"
#include "limbdark.h"
#include "tables.h"
#include "rotation.h"
#include "basis.h"

using namespace boost::math::quadrature;
using namespace utils;
using std::abs;
using std::max;
namespace py = pybind11;

template <class T>
class TransitInfo {

public:

    limbdark::GreensLimbDark<T> L;
    Vector<T> agol_c;
    T agol_norm;

    explicit TransitInfo(const int lmax, const Vector<double>& u) : L(lmax) {
        // Convert to Agol basis
        Vector<T> u_(lmax + 1);
        u_(0) = -1.0;
        u_.segment(1, lmax) = u.template cast<T>();
        agol_c = limbdark::computeC(u_);
        agol_norm = limbdark::normC(agol_c);
    }

};

template <class T>
inline T flux(const T& b, const T& r, TransitInfo<T>& I) {
    if (b < 1 + r) {
        I.L.compute(b, r);
        return I.L.S.dot(I.agol_c) * I.agol_norm;
    } else {
        return 1.0;
    }
}

template <class T>
T dfluxdb(int n, const T& b, const T& r, TransitInfo<T>& I) {
    Vector<T> a;
    T eps;
    if (n == 2) {
        a.resize(3);
        a << 1, -2, 1;
        eps = pow(mach_eps<T>(), 1. / 3.);
    } else if (n == 4){
        a.resize(5);
        a << 1, -4, 6, -4, 1;
        eps = pow(mach_eps<T>(), 1. / 6.);
    } else if (n == 6) {
        a.resize(7);
        a << 1, -6, 15, -20, 15, -6, 1;
        eps = pow(mach_eps<T>(), 1. / 9.);
    } else if (n == 8) {
       a.resize(9);
       a << 1, -8, 28, -56, 70, -56, 28, -8, 1;
       eps = pow(mach_eps<T>(), 1. / 12.);
    } else {
        throw std::invalid_argument( "Invalid order." );
    }
    T res = 0;
    int imax = (a.rows() - 1) / 2;
    for (int i = -imax, j = 0; i <= imax; ++i, ++j ) {
        res += a(j) * flux(abs(b + i * eps), r, I);
    }
    return res / (pow(eps, n));
}

template <class T>
T fluenceQuad(const T& expo, const T& b, const T& r, TransitInfo<T>& I) {
    auto f = [&r, &I](const T& b_) { return flux(abs(b_), r, I); };
    return gauss<T, 128>::integrate(f, b - 0.5 * expo, b + 0.5 * expo) / expo;
}


template <class T>
T fint(const int order, const T& t1, const T& t2, const T& b, const T& r, TransitInfo<T>& I) {
    T tavg = 0.5 * abs(t1 + t2);
    T tdif = (t2 - t1);
    T f = flux(tavg, r, I);
    if (order > 0) {
        f += (1 / 24.) * pow(tdif, 2) * dfluxdb(2, tavg, r, I);
        if (order > 2) {
            f += (1 / 1920.) * pow(tdif, 4) * dfluxdb(4, tavg, r, I);
            if (order > 4) {
                f += (1 / 322560.) * pow(tdif, 6) * dfluxdb(6, tavg, r, I);
                if (order > 6) {
                    f += (1 / 92897280.) * pow(tdif, 8) * dfluxdb(8, tavg, r, I);
                }
            }
        }
    }
    return f * tdif;
}


template <class T>
T fluenceTaylor(const int order, const T& expo, const T& b, const T& r, TransitInfo<T>& I) {

    // All possible limits of integration
    T e = 0.5 * expo;
    T P = 1 - r,
      Q = 1 + r,
      A = P - e,
      B = P + e,
      C = Q - e,
      D = Q + e,
      E = 2 * b - P,
      F = 2 * b - Q,
      Z = 0.0;
    std::vector<T> all_limits {A, B, C, D, E, F, P, Q, Z};

    // Identify and sort the relevant ones
    std::vector<T> limits;
    limits.push_back(b - e);
    for (auto lim : all_limits) {
        if ((lim > b - e) && (lim < b + e))
            limits.push_back(lim);
    }
    limits.push_back(b + e);
    std::sort(limits.begin() + 1, limits.end() - 1);

    // Compute the integrals
    T f = 0;
    T dt;
    for (size_t i = 0; i < limits.size() - 1; ++i) {
        f += fint(order, limits[i], limits[i + 1], b, r, I);
    }

    return f / expo;
}

Vector<double> computeFlux(const Vector<double>& b, const double& r, const Vector<double>& u) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);
    TransitInfo<double> dblI(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = flux(bi, r_, I);
    }

    return f.template cast<double>();

}

Vector<double> computeTaylorFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int order) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = fluenceTaylor(order, expo_, bi, r_, I);
    }

    return f.template cast<double>();

}

Vector<double> computeExactFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = fluenceQuad(expo_, bi, r_, I);
    }

    return f.template cast<double>();

}

Vector<double> computeLeftRiemannFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int ndiv) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / ndiv;
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_;
        for (int j = 0; j < ndiv; ++j) {
            f(i) += flux(abs(b0 + db * j), r_, I);
        }
        f(i) /= ndiv;
    }

    return f.template cast<double>();

}

Vector<double> computeRiemannFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int ndiv) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / (ndiv + 1);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_ + db;
        for (int j = 0; j < ndiv; ++j) {
            f(i) += flux(abs(b0 + db * j), r_, I);
        }
        f(i) /= ndiv;
    }

    return f.template cast<double>();

}

Vector<double> computeTrapezoidFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int ndiv) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / ndiv;
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_;
        for (int j = 0; j < ndiv; ++j) {
            f(i) += 0.5 * (flux(abs(b0 + db * j), r_, I) + flux(abs(b0 + db * (j + 1)), r_, I));
        }
        f(i) /= ndiv;
    }

    return f.template cast<double>();

}

Vector<double> computeSimpsonFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, int ndiv) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Vector<T> f(npts);
    TransitInfo<T> I(lmax, u);

    if (ndiv % 2 != 0) ndiv += 1;

    // Run!
    T bi;
    T r_ = T(r);
    T expo_ = T(expo);
    T db = expo_ / ndiv;
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i) = 0.0;
        T b0 = bi - 0.5 * expo_;
        for (int j = 0; j <= ndiv; ++j) {
            T f0 = flux(abs(b0 + db * j), r_, I);
            if (j == 0 || j == ndiv)
                f(i) += f0;
            else
                f(i) += (2 + 2 * (j % 2)) * f0;
            // f(i) += 0.5 * (flux(abs(b0 + db * j), r_, I) + flux(abs(b0 + db * (j + 1)), r_, I));
        }
        f(i) /= 3 * ndiv;
    }

    return f.template cast<double>();

}

Vector<double> computePhaseCurveFluence(const Vector<double>& time, Vector<double>& y,
    UnitVector<double>& axis, const double& per, const double& t0, const double& theta0,
    const double& expo) {

    // Initialize stuff
    int N = y.size();
    int lmax = int(sqrt(N) - 1);
    basis::Basis<double> B(lmax);
    rotation::Wigner<Vector<double>> W(lmax, 1, y, axis);
    W.update();

    // Frame transforms
    VectorT<double> P(N);
    Vector<double> Q(N), QRev(N);
    for (int l = 0; l < lmax + 1; l++) {
        P.segment(l * l, 2 * l + 1) = B.rTA1.segment(l * l, 2 * l + 1) * W.RZetaInv[l];
        Q.segment(l * l, 2 * l + 1) = W.RZeta[l] * y.segment(l * l, 2 * l + 1);
    }
    Vector<double> RQ;

    // Theta vector
    Vector<double> theta = Vector<double>::Ones(time.size()) * theta0 + (2 * pi<double>() / per) * (time - Vector<double>::Ones(time.size()) * t0);

    // Compute the phase curve
    Vector<double> f(time.size());
    f.setZero();
    for (int i = 0; i < time.size(); ++i) {
        
        W.rotatez(theta(i), expo, per, Q, RQ);
        f(i) = P * RQ;


        // DEBUG: Numerical using centered riemann
        /*
        int n = 50;
        double expo_theta = 2 * pi<double>() / per * expo;
        double dt = expo_theta / (n + 1);
        double t0 = theta(i) - 0.5 * expo_theta + dt;
        f(i) = 0;
        for (int j = 0; j < n; ++j) {
            W.rotatez(cos(t0 + j * dt), sin(t0 + j * dt), Q, RQ);
            f(i) += P * RQ;
        }
        f(i) /= n;
        */

    }

    return f;
}


PYBIND11_MODULE(fluence, m) {

    /*
    m.def("flux", &computeFlux);

    m.def("taylor_fluence", &computeTaylorFluence);

    m.def("exact_fluence", &computeExactFluence);

    m.def("left_riemann_fluence", &computeLeftRiemannFluence);

    m.def("riemann_fluence", &computeRiemannFluence);

    m.def("trapezoid_fluence", &computeTrapezoidFluence);

    m.def("simpson_fluence", &computeSimpsonFluence);
    */

    m.def("phase_curve_fluence", &computePhaseCurveFluence);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
