/**
Defines the surface map class.

*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "constants.h"
#include "rotation.h"
#include "basis.h"
#include "solver.h"
#include "numeric.h"
#include "errors.h"

// Multiprecision
#ifndef STARRY_MP_DIGITS
#define STARRY_MP_DIGITS    32
#endif
#include <boost/multiprecision/cpp_dec_float.hpp>
typedef boost::multiprecision::cpp_dec_float<STARRY_MP_DIGITS> mp_backend;
typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off> bigdouble;

// Shorthand
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;

namespace maps {

    // Some default unit vectors
    UnitVector<double> xhat({1, 0, 0});
    UnitVector<double> yhat({0, 1, 0});
    UnitVector<double> zhat({0, 0, 1});


    // Constant matrices/vectors
    class Constants {

        public:

            int lmax;
            Eigen::SparseMatrix<double> A1;
            Eigen::SparseMatrix<double> A;
            VectorT<double> rTA1;
            VectorT<double> rT;

            // Constructor: compute the matrices
            Constants(int lmax) : lmax(lmax) {
                basis::computeA1(lmax, A1);
                basis::computeA(lmax, A1, A);
                solver::computerT(lmax, rT);
                rTA1 = rT * A1;
            }

    };


    // ****************************
    // ----------------------------
    //
    // The surface map vector class
    //
    // ----------------------------
    // ****************************
    template <class T>
    class Map {

        protected:

            Vector<T> basis;
            bool needs_update;

            // Temporary variables
            Vector<T> tmpvec;
            T tmpscalar;
            T tmpu1, tmpu2, tmpu3;
            Vector<T> ARRy;

            // Private methods
            void apply_rotation(UnitVector<T>& axis, T costheta, T sintheta,
                                Vector<T>& yin, Vector<T>& yout);

        public:

            // The map vectors
            Vector<T> y;
            Vector<T> p;
            Vector<T> g;

            // Map order
            int N;
            int lmax;

            // Rotation matrices
            rotation::Wigner<T> R;

            // Constant matrices
            Constants C;

            // Greens data
            Vector<bigdouble> mpVec;
            solver::Greens<bigdouble> mpG;
            solver::Greens<T> G;

            // Multiprecision flag
            bool use_mp;

            // Constructor: initialize map to zeros
            Map(int lmax=2) :
                  lmax(lmax), R(lmax), C(lmax),
                  mpG(lmax, false),
                  G(lmax, true) {
                N = (lmax + 1) * (lmax + 1);
                y = Vector<T>::Zero(N);
                p = Vector<T>::Zero(N);
                g = Vector<T>::Zero(N);
                tmpvec = Vector<T>::Zero(N);
                ARRy = Vector<T>::Zero(N);
                mpVec = Vector<bigdouble>::Zero(N);
                tmpscalar = NAN;
                tmpu1 = 0;
                tmpu2 = 0;
                tmpu3 = 0;
                basis.resize(N, 1);
                use_mp = false;
                update(true);
            }

            // Public methods
            T evaluate(UnitVector<T>& axis=yhat, T theta=0, T x0=0, T y0=0);
            void rotate(UnitVector<T>& axis, T theta, Vector<T>& yin,
                        Vector<T>& yout);
            void rotate(UnitVector<T>& axis, T costheta, T sintheta,
                        Vector<T>& yin, Vector<T>& yout);
            void rotate(UnitVector<T>& axis, T theta);
            void rotate(UnitVector<T>& axis, T costheta, T sintheta);
            void update(bool force=false);
            void random(double beta=0);
            void set_coeff(int l, int m, T coeff);
            T get_coeff(int l, int m);
            void reset();
            T flux(UnitVector<T>& axis=yhat, T theta=0,
                   T xo=0, T yo=0, T ro=0,
                   bool numerical=false, double tol=1e-4);
            std::string repr();

    };

    // Rotate a map `yin` and store the result in `yout`
    template <class T>
    void Map<T>::apply_rotation(UnitVector<T>& axis, T costheta, T sintheta,
                                Vector<T>& yin, Vector<T>& yout) {

        // Compute the rotation matrix R
        rotation::computeR(lmax, axis, costheta, sintheta, R.Complex, R.Real);

        // Dot R in, order by order
        for (int l = 0; l < lmax + 1; l++) {
            yout.segment(l * l, 2 * l + 1) = R.Real[l] *
                                             yin.segment(l * l, 2 * l + 1);
        }

        return;
    }

    // Update the maps after the coefficients changed
    // or after a base rotation was applied
    template <class T>
    void Map<T>::update(bool force) {
        if (force || needs_update) {
            p = C.A1 * y;
            g = C.A * y;
            tmpscalar = NAN;
            tmpu1 = 0;
            tmpu2 = 0;
            tmpu3 = 0;
            tmpvec = Vector<T>::Zero(N);
            needs_update = false;
        }
    }


    // Evaluate our map at a given (x0, y0) coordinate
    template <class T>
    T Map<T>::evaluate(UnitVector<T>& axis, T theta, T x0, T y0) {

        // Update the maps if necessary
        update();

        // Get the polynomial map
        Vector<T>* ptrmap;

        if (theta == 0) {
            // We will use this.p
            ptrmap = &p;
        } else if ((theta == tmpscalar) && (axis(0) == tmpu1) && (axis(1) == tmpu2) && (axis(2) == tmpu3)) {
            // We will use this.tmpvec, which we computed last time around
            ptrmap = &tmpvec;
        } else {
            // Rotate the map into view
            rotate(axis, theta, y, tmpvec);
            tmpvec = C.A1 * tmpvec;
            ptrmap = &tmpvec;
        }

        // Save this value of theta so we don't have
        // to keep rotating the map when we vectorize
        // this function!
        tmpscalar = theta;
        tmpu1 = axis(0);
        tmpu2 = axis(1);
        tmpu3 = axis(2);

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN;

        int l, m, mu, nu, n = 0;
        T z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Compute the polynomial basis where it is needed
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (std::abs((*ptrmap)(n)) < STARRY_MAP_TOLERANCE) {
                    basis(n) = 0;
                } else {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0)
                        basis(n) = pow(x0, mu / 2) * pow(y0, nu / 2);
                    else
                        basis(n) = pow(x0, (mu - 1) / 2) *
                                   pow(y0, (nu - 1) / 2) * z0;
                }
                n++;
            }
        }

        // Dot the coefficients in to our polynomial map
        return (*ptrmap).dot(basis);

    }

    // Shortcut to rotate the base map in-place given `theta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& axis, T theta) {
        apply_rotation(axis, cos(theta), sin(theta), y, y);
        needs_update = true;
    }

    // Shortcut to rotate the base map in-place given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& axis, T costheta, T sintheta) {
        apply_rotation(axis, costheta, sintheta, y, y);
        needs_update = true;
    }

    // Shortcut to rotate an arbitrary map given `theta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& axis, T theta,
                        Vector<T>& yin, Vector<T>& yout) {
        apply_rotation(axis, cos(theta), sin(theta), yin, yout);
    }

    // Shortcut to rotate an arbitrary map given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& axis, T costheta, T sintheta,
                        Vector<T>& yin, Vector<T>& yout) {
        apply_rotation(axis, costheta, sintheta, yin, yout);
    }

    // Compute the total flux during or outside of an occultation
    template <class T>
    T Map<T>::flux(UnitVector<T>& axis, T theta, T xo, T yo, T ro,
                   bool numerical, double tol) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // Pointer to the map we're integrating
        // (defaults to the base map)
        Vector<T>* ptry = &y;

        // Rotate the map into view if necessary and update our pointer
        if (theta != 0) {
            rotate(axis, theta, (*ptry), tmpvec);
            ptry = &tmpvec;
        }

        // Compute it numerically?
        if (numerical) {
            tmpvec = C.A1 * (*ptry);
            return numeric::flux(xo, yo, ro, lmax, tmpvec, tol);
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            return C.rTA1 * (*ptry);

        // Occultation
        } else {

            // Align occultor with the +y axis if necessary
            if ((b > 0) && (xo != 0)) {
                rotate(zhat, yo / b, xo / b, (*ptry), tmpvec);
                ptry = &tmpvec;
            }

            // Perform the rotation + change of basis
            ARRy = C.A * (*ptry);

            if (use_mp) {

                // Compute sT using Boost multiprecision
                // This is *much* slower (~20x) than using doubles.
                // TODO: Investigate how to get this to work with autodiff!
                mpVec = ARRy.template cast<bigdouble>();
                bigdouble mpb = b;
                bigdouble mpro = ro;
                solver::computesT<bigdouble>(mpG, mpb, mpro, mpVec);

                // Dot the result in
                bigdouble tmp = mpG.sT * mpVec;
                return (T) tmp;

            } else {

                // Compute the sT vector
                solver::computesT<T>(G, b, ro, ARRy);

                // Dot the result in and we're done
                return G.sT * ARRy;
            }

        }

    }

    // Set the (l, m) coefficient
    template <class T>
    void Map<T>::set_coeff(int l, int m, T coeff) {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l)) {
            int n = l * l + l + m;
            y(n) = coeff;
            needs_update = true;
        } else throw errors::BadLM();
    }

    // Get the (l, m) coefficient
    template <class T>
    T Map<T>::get_coeff(int l, int m) {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
            return y(l * l + l + m);
        else throw errors::BadLM();
    }

    // Reset the map
    template <class T>
    void Map<T>::reset() {
        y.setZero(N);
        needs_update = true;
    }

    // Generate a random map with a given power spectrum power index `beta`
    template <class T>
    void Map<T>::random(double beta) {
        int l, m, n;
        double norm;
        Vector<double> coeffs;
        set_coeff(0, 0, 1.);
        for (l = 1; l < lmax + 1; l++) {
            coeffs = Vector<double>::Random(2 * l + 1);
            norm = pow(l, beta) / coeffs.squaredNorm();
            n = 0;
            for (m = -l; m < l + 1; m++) {
                set_coeff(l, m, coeffs(n) * norm);
                n++;
            }
        }
    }

    // Return a human-readable map string
    template <class T>
    std::string Map<T>::repr() {
        int n = 0;
        int nterms = 0;
        char buf[30];
        std::ostringstream os;
        os << "<STARRY Map: ";
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < l + 1; m++) {
                if (std::abs(y(n)) > STARRY_MAP_TOLERANCE){
                    // Separator
                    if ((nterms > 0) && (y(n) > 0)) {
                        os << " + ";
                    } else if ((nterms > 0) && (y(n) < 0)){
                        os << " - ";
                    } else if ((nterms == 0) && (y(n) < 0)){
                        os << "-";
                    }
                    // Term
                    if ((y(n) == 1) || (y(n) == -1)) {
                        sprintf(buf, "Y_{%d,%d}", l, m);
                        os << buf;
                    } else if (fmod(std::abs(y(n)), 1) < STARRY_MAP_TOLERANCE) {
                        sprintf(buf, "%d Y_{%d,%d}", (int)std::abs(y(n)), l, m);
                        os << buf;
                    } else if (fmod(std::abs(y(n)), 1) >= 0.01) {
                        sprintf(buf, "%.2f Y_{%d,%d}", std::abs(y(n)), l, m);
                        os << buf;
                    } else {
                        sprintf(buf, "%.2e Y_{%d,%d}", std::abs(y(n)), l, m);
                        os << buf;
                    }
                    nterms++;
                }
                n++;
            }
        }
        if (nterms == 0)
            os << "Null";
        os << ">";
        return std::string(os.str());
    }


    // ****************************
    // ----------------------------
    //
    // The limb-darkened map class
    //
    // ----------------------------
    // ****************************
    template <class T>
    class LimbDarkenedMap {

        protected:

            Vector<T> basis;
            bool needs_update;
            T ld_flux;

            // Temporary variables
            Vector<T> ARRy;
            Vector<T> tmpvec;

        public:

            // The map vectors
            Vector<T> y;
            Vector<T> p;
            Vector<T> g;
            Vector<T> u;

            // Map order
            int N;
            int lmax;

            // Constant matrices
            Constants C;

            // Greens data
            Vector<bigdouble> mpVec;
            solver::Greens<bigdouble> mpG;
            solver::Greens<T> G;

            // Multiprecision flag
            bool use_mp;

            // Constructor: initialize map to zeros
            LimbDarkenedMap(int lmax=2) :
                  lmax(lmax), C(lmax),
                  mpG(lmax, false),
                  G(lmax, true) {
                N = (lmax + 1) * (lmax + 1);
                y = Vector<T>::Zero(N);
                p = Vector<T>::Zero(N);
                g = Vector<T>::Zero(N);
                u = Vector<T>::Zero(lmax + 1);
                tmpvec = Vector<T>::Zero(N);
                ARRy = Vector<T>::Zero(N);
                mpVec = Vector<bigdouble>::Zero(N);
                basis.resize(N, 1);
                use_mp = false;
                reset();
                update(true);
            }

            // Public methods
            T evaluate(T x0=0, T y0=0);
            void update(bool force=false);
            void set_coeff(int l, T coeff);
            T get_coeff(int l);
            void reset();
            T flux(T xo=0, T yo=0, T ro=0, bool numerical=false, double tol=1e-4);
            std::string repr();

    };

    // Update the maps after the coefficients changed
    // or after a base rotation was applied
    template <class T>
    void LimbDarkenedMap<T>::update(bool force) {
        if (force || needs_update) {
            p = C.A1 * y;
            g = C.A * y;
            needs_update = false;
        }
    }

    // Evaluate our map at a given (x0, y0) coordinate
    template <class T>
    T LimbDarkenedMap<T>::evaluate(T x0, T y0) {

        // Update the map if necessary
        update();

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN;

        int l, m, mu, nu, n = 0;
        T z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Compute the polynomial basis where it is needed
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (std::abs(p(n)) < STARRY_MAP_TOLERANCE) {
                    basis(n) = 0;
                } else {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0)
                        basis(n) = pow(x0, mu / 2) * pow(y0, nu / 2);
                    else
                        basis(n) = pow(x0, (mu - 1) / 2) *
                                   pow(y0, (nu - 1) / 2) * z0;
                }
                n++;
            }
        }

        // Dot the coefficients in to our polynomial map
        return p.dot(basis);

    }

    // Compute the total flux during or outside of an occultation
    template <class T>
    T LimbDarkenedMap<T>::flux(T xo, T yo, T ro, bool numerical, double tol) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // If we're doing quadratic limb darkening, let's skip all the overhead
        if ((lmax <= 2) && (ro < 1) && (!numerical) && (!use_mp)) {
            if ((b >= 1 + ro) || (ro == 0))
                return ld_flux;
            else
                return solver::QuadLimbDark(G, b, ro, g(0), g(2), g(8));
        }

        // Compute it numerically?
        if (numerical) {
            tmpvec = C.A1 * y;
            return numeric::flux(xo, yo, ro, lmax, tmpvec, tol);
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            return C.rTA1 * y;

        // Occultation
        } else {

            // Perform the change of basis
            ARRy = C.A * y;

            if (use_mp) {

                // Compute sT using Boost multiprecision
                // This is *much* slower (~20x) than using doubles.
                // TODO: Investigate how to get this to work with autodiff!
                mpVec = ARRy.template cast<bigdouble>();
                bigdouble mpb = b;
                bigdouble mpro = ro;
                solver::computesT<bigdouble>(mpG, mpb, mpro, mpVec);

                // Dot the result in
                bigdouble tmp = mpG.sT * mpVec;
                return (T) tmp;

            } else {

                // Compute the sT vector
                solver::computesT<T>(G, b, ro, ARRy);

                // Dot the result in and we're done
                return G.sT * ARRy;
            }

        }

    }

    // Set a limb darkening coefficient
    template <class T>
    void LimbDarkenedMap<T>::set_coeff(int l, T u_l) {
        if ((l <= 0) || (l > lmax)) {
            throw errors::BadIndex();
        } else if (l > 2) {
            // TODO! Implement higher order limb darkening.
            throw errors::LimbDark();
        }

        // Set the limb darkening coefficient
        u(l) = u_l;

        // Update the map vector
        y.setZero(N);
        if (lmax == 0) {
            y(0) = 2 * sqrt(M_PI);
        } else if (lmax == 1) {
            y(0) = 2 * sqrt(M_PI) / 3. * (3 - 3 * u(1));
            y(2) = 2 * sqrt(M_PI / 3.) * u(1);
        } else {
            y(0) = 2 * sqrt(M_PI) / 3. * (3 - 3 * u(1) - 4 * u(2));
            y(2) = 2 * sqrt(M_PI / 3.) * (u(1) + 2 * u(2));
            y(6) = -4. / 3. * sqrt(M_PI / 5) * u(2);
        }

        // Pre-compute the greens polynomials so we can
        // breeze through the flux calculation
        g = C.A * y;
        ld_flux = G.pi * g(0) + 2. * G.pi / 3. * g(2) + G.pi_over_2 * g(8);
        needs_update = true;
    }

    // Get a limb darkening coefficient
    template <class T>
    T LimbDarkenedMap<T>::get_coeff(int l) {
        if ((l <= 0) || (l > lmax)) {
            throw errors::BadIndex();
        } else if (l > 2) {
            // TODO! Implement higher order limb darkening.
            throw errors::LimbDark();
        } else {
            return u(l);
        }
    }

    // Reset the map
    template <class T>
    void LimbDarkenedMap<T>::reset() {
        u.setZero(lmax + 1);
        y.setZero(N);
        y(0) = 2 * sqrt(M_PI);
        g = C.A * y;
        ld_flux = G.pi * g(0);
        needs_update = true;
    }

    // Return a human-readable map string
    template <class T>
    std::string LimbDarkenedMap<T>::repr() {
        int n = 0;
        int nterms = 0;
        char buf[30];
        std::ostringstream os;
        os << "<STARRY LimbDarkenedMap: ";
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < l + 1; m++) {
                if (std::abs(y(n)) > STARRY_MAP_TOLERANCE){
                    // Separator
                    if ((nterms > 0) && (y(n) > 0)) {
                        os << " + ";
                    } else if ((nterms > 0) && (y(n) < 0)){
                        os << " - ";
                    } else if ((nterms == 0) && (y(n) < 0)){
                        os << "-";
                    }
                    // Term
                    if ((y(n) == 1) || (y(n) == -1)) {
                        sprintf(buf, "Y_{%d,%d}", l, m);
                        os << buf;
                    } else if (fmod(std::abs(y(n)), 1) < STARRY_MAP_TOLERANCE) {
                        sprintf(buf, "%d Y_{%d,%d}", (int)std::abs(y(n)), l, m);
                        os << buf;
                    } else if (fmod(std::abs(y(n)), 1) >= 0.01) {
                        sprintf(buf, "%.2f Y_{%d,%d}", std::abs(y(n)), l, m);
                        os << buf;
                    } else {
                        sprintf(buf, "%.2e Y_{%d,%d}", std::abs(y(n)), l, m);
                        os << buf;
                    }
                    nterms++;
                }
                n++;
            }
        }
        if (nterms == 0)
            os << "Null";
        os << ">";
        return std::string(os.str());
    }

}; // namespace maps

#endif
