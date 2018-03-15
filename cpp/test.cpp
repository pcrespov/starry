#define STARRY_NO_AUTODIFF 1
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include "ellip.h"
#include "maps.h"
#include "basis.h"
#include "fact.h"
#include "sqrtint.h"
#include "rotation.h"
#include "solver.h"

using namespace std;

int main() {

    // Generate a map
    maps::Map<double> y = maps::Map<double>(2);
    y.set_coeff(1, -1, 1);
    cout << y.repr() << endl;

    // Compute the occultation flux
    int npts = 20;
    double r = 0.25;
    double y0 = 0.25;
    double diff;
    Eigen::VectorXd x0 = Eigen::VectorXd::LinSpaced(npts, -1.5, 1.5);
    Eigen::VectorXd flux = Eigen::VectorXd::Zero(npts);
    Eigen::VectorXd numflux = Eigen::VectorXd::Zero(npts);

    cout << endl;
    cout << setw(12) << "Analytic" << "     "
         << setw(12) << "Numerical" << "    "
         << setw(12) << "Difference" << endl;
         cout << setw(12) << "--------" << "     "
              << setw(12) << "---------" << "    "
              << setw(12) << "----------" << endl;
    for (int i = 0; i < npts; i ++) {
        flux(i) = y.flux(maps::yhat, 0, x0(i), y0, r);
        numflux(i) = y.flux(maps::yhat, 0, x0(i), y0, r, true, 1e-5);
        diff = (flux(i) - numflux(i));
        cout << setw(12) << flux(i) << "     "
             << setw(12) << numflux(i) << "    "
             << setw(12) << diff << endl;
    }

    return 0;
}
