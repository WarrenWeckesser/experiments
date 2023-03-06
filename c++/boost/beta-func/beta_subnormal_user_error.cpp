
#include <iostream>
#include <iomanip>
#include <cfenv>

//#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions.hpp>

//using namespace std;
using std::cout;  using std::endl; using std::cerr;

namespace boost {
    namespace math {
        namespace policies {
            template <class T>
            T user_overflow_error(const char* function, const char* message, const T& val)
            {
                cerr << "Overflow Error!" << endl;
                return std::numeric_limits<T>::quiet_NaN();
            }
        } // namespace policies
    } // namespace math
} // namespace boost



namespace mymath{

using namespace boost::math::policies;

typedef policy<
   overflow_error<user_error>
> user_error_policy;

BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(user_error_policy)

} // close unnamed namespace


void show_fp_exception_flags()
{
    if (std::fetestexcept(FE_DIVBYZERO)) {
        cout << " FE_DIVBYZERO";
    }
    // FE_INEXACT is common and not interesting.
    // if (std::fetestexcept(FE_INEXACT)) {
    //     cout << " FE_INEXACT";
    // }
    if (std::fetestexcept(FE_INVALID)) {
        cout << " FE_INVALID";
    }
    if (std::fetestexcept(FE_OVERFLOW)) {
        cout << " FE_OVERFLOW";
    }
    if (std::fetestexcept(FE_UNDERFLOW)) {
        cout << " FE_UNDERFLOW";
    }
    cout << endl;
}

int main(int argc, char *argv[])
{
    double a = 12.5;
    double b = 1e-320;

    std::feclearexcept(FE_ALL_EXCEPT);

    double x = mymath::beta(a, b);

    show_fp_exception_flags();

    std::cout << std::scientific << std::setw(24)
              << std::setprecision(17) << x << std::endl;

    return 0;
}
