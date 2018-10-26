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

using namespace boost::math::quadrature;
using namespace utils;
using std::abs;
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
T fluenceTaylor(const int order, const int subdiv, const T& expo, const T& b, const T& r, TransitInfo<T>& I) {

    // Boundaries
    T e = 0.5 * expo;
    T P = 1 - r;
    T Q = 1 + r;
    T A = P - e;
    T B = P + e;
    T C = Q - e;
    T D = Q + e;

    /*
    // Limits of integration
    std::vector<T> t;

    // First boundary
    t.push_back(b - e);

    // Cases
    if (B < C) {
        // expo < 2 * r
        if ((b >= A) && (b <= B)) {
            if (b < P) {
                t.push_back(A);
                t.push_back(2 * b - P);
                t.push_back(P);
            } else {
                t.push_back(P);
                t.push_back(2 * b - P);
                t.push_back(B);
            }
        } else if ((b >= C) && (b <= D)) {
            if (b < Q) {
                t.push_back(C);
                t.push_back(2 * b - Q);
                t.push_back(Q);
            } else {
                t.push_back(Q);
                t.push_back(2 * b - Q);
                t.push_back(D);
            }
        }
    } else if (B < Q) {
        // 2 * r < expo < 4 * r
        // TODO
    } else {
        // expo > 4 * r
        // TODO
    }

    // Last boundary
    t.push_back(b + e);
    */

    // DEBUG
    std::vector<T> t {b - e, A, B, C, D, P, Q, 2 * b - P, 2 * b - Q, b + e};
    std::sort(t.begin(), t.end());

    // Compute the integrals
    T f = 0;
    T dt;
    for (size_t i = 0; i < t.size() - 1; ++i) {

        if ((t[i] < b - e) || (t[i + 1] > b + e))
            continue;

        dt = (t[i + 1] - t[i]) / subdiv;
        for (int j = 0; j < subdiv; ++j) {
            f += fint(order, t[i] + j * dt, t[i] + (j + 1) * dt, b, r, I);
        }
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

Vector<double> computeFluence(const Vector<double>& b, const double& r, const Vector<double>& u, const double& expo, const int order=99, const int subdiv=1) {

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
        if (order < 8)
            f(i) = fluenceTaylor(order, subdiv, expo_, bi, r_, I);
        else
            f(i) = fluenceQuad(expo_, bi, r_, I);
    }

    return f.template cast<double>();

}

PYBIND11_MODULE(fluence, m) {

    m.def("flux", &computeFlux);

    m.def("fluence", &computeFluence);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
