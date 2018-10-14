#define STARRY_IJ_MAX_ITER                 1600
#define STARRY_NMULTI                      256

#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"
#include "limbdark.h"
#include "tables.h"
#include <unsupported/Eigen/NumericalDiff>
#include <fstream>
#include <stdexcept>

using namespace utils;
using std::abs;

template <class T>
inline T flux(const T& b, const T& r, limbdark::GreensLimbDark<T>& L, const Vector<T>& agol_c, const T& agol_norm) {
    if (b < 1 + r) {
        L.compute(b, r);
        return L.S.dot(agol_c) * agol_norm;
    } else {
        return 1.0;
    }
}

template <class T>
T dfluxdb(int n, const T& b, const T& r, limbdark::GreensLimbDark<T>& L, const Vector<T>& agol_c, const T& agol_norm) {
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
    } else {
        throw std::invalid_argument( "Invalid order." );
    }

    T res = 0;
    int imax = (a.rows() - 1) / 2;
    for (int i = -imax, j = 0; i <= imax; ++i, ++j ) {
        res += a(j) * flux(abs(b + i * eps), r, L, agol_c, agol_norm);
    }
    return res / (pow(eps, n));
}

int main() {

    using T = Multi;
    int npts = 99;
    int lmax = 2;
    limbdark::GreensLimbDark<T> L(lmax);
    Vector<T> u(lmax + 1);
    u(0) = -1;
    Vector<T> agol_c;
    T agol_norm;
    Vector<T> b = Vector<T>::LinSpaced(npts, -1.2, 1.2);
    Matrix<T> f(npts, 5);
    T r, eps;

    std::cout << pow(mach_eps<T>(), 1. / 3.) << ", "
              << pow(mach_eps<T>(), 1. / 6.) << ", "
              << pow(mach_eps<T>(), 1. / 9.) << std::endl;

    // Data
    u(1) = 0.4;
    u(2) = 0.26;
    r = 0.1;
    agol_c = limbdark::computeC(u);
    agol_norm = limbdark::normC(agol_c);

    // Run
    for (int i = 0; i < npts; ++i) {
        f(i, 0) = b(i);
        f(i, 1) = flux(abs(b(i)), r, L, agol_c, agol_norm);
        f(i, 2) = dfluxdb(2, abs(b(i)), r, L, agol_c, agol_norm);
        f(i, 3) = dfluxdb(4, abs(b(i)), r, L, agol_c, agol_norm);
        f(i, 4) = dfluxdb(6, abs(b(i)), r, L, agol_c, agol_norm);
    }

    // Save
    std::ofstream file("test.txt");
    if (file.is_open()) {
      file << std::setprecision(16) << f << '\n';
    }

}
