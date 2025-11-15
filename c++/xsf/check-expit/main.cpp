#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <xsf/log_exp.h>


void clear_fp_stuff()
{
    errno = 0;
    std::feclearexcept(FE_ALL_EXCEPT);
}

void show_fp_stuff(const char *label)
{
    std::cout << label << std::endl;
    std::cout << "errno = " << errno << "   " << std::strerror(errno) << std::endl;

    if (std::fetestexcept(FE_DIVBYZERO))
        std::cout << "FE_DIVBYZERO (pole error) reported\n";
    if (std::fetestexcept(FE_OVERFLOW))
        std::cout << "FE_OVERFLOW reported\n";
    if (std::fetestexcept(FE_UNDERFLOW))
        std::cout << "FE_UNDERFLOW reported\n";
    if (std::fetestexcept(FE_INEXACT))
        std::cout << "FE_INEXACT reported\n";
    if (std::fetestexcept(FE_INVALID))
        std::cout << "FE_INVALID reported\n";
}

template<typename T>
T my_logistic(T x)
{
    if (x > 0) {
        T e;
        constexpr double logmax = -std::log(std::numeric_limits<T>::epsilon()/2);
        if (x > logmax) {
            e = std::numeric_limits<T>::min();
        }
        else {
            e = std::exp(-x);
        }
        return 1/(1 + e);
    }
    else {
        T t = std::exp(x);
        return t/(1 + t);
    }
}

int main()
{
    std::cout << "MATH_ERRNO is "
              << (math_errhandling & MATH_ERRNO ? "set" : "not set") << '\n'
              << "MATH_ERREXCEPT is "
              << (math_errhandling & MATH_ERREXCEPT ? "set" : "not set") << '\n';

    double x = 36.736;
    //double x = -std::log(std::numeric_limits<double>::epsilon());
    //x = std::nextafter(x, INFINITY);
    //x = std::nextafter(x, INFINITY);


    clear_fp_stuff();

    show_fp_stuff("initial");

    double y1 = xsf::expit(x);
    show_fp_stuff("\nafter xsf::expit");

    clear_fp_stuff();
    double y2 = 1/(1 + std::exp(-x));
    show_fp_stuff("\nafter 1/(1+exp(-x))");

    clear_fp_stuff();
    double y3 = my_logistic(x);
    show_fp_stuff("\nafter my_logistic(x)");

    std::cout << std::endl;
    std::cout << "x  = " << std::setprecision(17) << x << std::endl;
    std::cout << "y1 = " << std::setprecision(17) << y1 << std::endl;
    std::cout << "y2 = " << std::setprecision(17) << y2 << std::endl;
    std::cout << "y3 = " << std::setprecision(17) << y3 << std::endl;

    /*
    clear_fp_stuff();
    double e = -std::log(std::numeric_limits<double>::min());
    //e = std::nextafter(e, 1.0);
    double y = 1/(1 + e);
    show_fp_stuff("\nexperiment");
    */
}
