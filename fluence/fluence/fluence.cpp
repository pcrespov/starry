#include <boost/math/quadrature/gauss.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
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
T numFluence(const T& exp, const T& b, const T& r, TransitInfo<T>& I) {
    auto f = [&r, &I](const T& b_) { return flux(abs(b_), r, I); };
    return gauss<T, 20>::integrate(f, b - 0.5 * exp, b + 0.5 * exp) / exp;
}

Matrix<double> compute(const Vector<double>& b, const double& r, const Vector<double>& u, const double& exp) {

    using T = Multi;
    int npts = b.rows();
    int lmax = u.rows();
    Matrix<T> f(npts, 7);
    TransitInfo<T> I(lmax, u);
    TransitInfo<double> dblI(lmax, u);

    /*
    std::cout << pow(mach_eps<T>(), 1. / 3.) << ", "
              << pow(mach_eps<T>(), 1. / 6.) << ", "
              << pow(mach_eps<T>(), 1. / 9.) << std::endl;
    */



    // Run!
    T bi;
    T r_ = T(r);
    for (int i = 0; i < npts; ++i) {
        bi = T(abs(b(i)));
        f(i, 0) = flux(bi, r_, I);
        f(i, 1) = numFluence(T(exp), bi, r_, I);
        f(i, 2) = dfluxdb(2, bi, r_, I);
        f(i, 3) = dfluxdb(4, bi, r_, I);
        f(i, 4) = dfluxdb(6, bi, r_, I);
        f(i, 5) = dfluxdb(8, bi, r_, I);
    }

    return f.template cast<double>();

}

PYBIND11_MODULE(fluence, m) {

    m.def("compute", &compute);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
