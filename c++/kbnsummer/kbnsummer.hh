#include <cmath>
#include <complex>
#include <cstdio>

template<typename T>
struct is_complex_t : public std::false_type {};

template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

//
// KBNSummer implements Kahan–Babuška-Neumaier summation.
//
// It is initialized to 0.0 when constructed.
// Values are added to the summer with the += operator.
// Cast to T is implemented, so this works:
//
//    double foo(x, y, z, a, b, c) {
//        KBNSummer<double> sum;
//        [... code that uses sum...]
//        return sum;  // Return the sum as a double.
//
// The template parameter T is intended to be a real or complex
// floating point type.
//

template<typename T>
class KBNSummer {

    // _acc is the accumulator of the values added so far.
    // _c is the compensation computed so far.
    // The final result of the values added so far is _acc + _c.
    T _acc;
    T _c;

    public:

    KBNSummer() {
        _acc = static_cast<T>(0);
        _c = static_cast<T>(0);
    }

    T acc() {
        return _acc;
    }

    T c() {
        return _c;
    }

    KBNSummer& operator+=(const T x) {
        T t = _acc + x;
        if constexpr (is_complex_t<T>::value) {
            // TODO: Clean up repeated code.
            if (std::abs(_acc.real()) >= std::abs(x.real())) {
                _c.real(_c.real() + ((_acc.real() - t.real()) + x.real()));
            }
            else {
                _c.real(_c.real() + ((x.real() - t.real()) + _acc.real()));
            }
            if (std::abs(_acc.imag()) >= std::abs(x.imag())) {
                _c.imag(_c.imag() + ((_acc.imag() - t.imag()) + x.imag()));
            }
            else {
                _c.imag(_c.imag() + ((x.imag() - t.imag()) + _acc.imag()));
            }
        }
        else {
            if (std::abs(_acc) >= std::abs(x)) {
                _c += (_acc - t) + x;
            }
            else {
                _c += (x - t) + _acc;
            }
        }
        _acc = t;
        return *this;
    }

    operator T() const {
        return _acc + _c;
    }

    T sum() {
        return _acc + _c;
    }
};
