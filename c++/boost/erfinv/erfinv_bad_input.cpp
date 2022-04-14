
#include <iostream>
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/erf.hpp>

using namespace std;

//using namespace boost::math::policies;

//using boost::math::policies::policy;
// Types of error whose action can be altered by policies:.
//using boost::math::policies::evaluation_error;
//using boost::math::policies::domain_error;
//using boost::math::policies::overflow_error;
//using boost::math::policies::underflow_error;
//using boost::math::policies::pole_error;
// Actions on error (in enum error_policy_type):
//using boost::math::policies::errno_on_error;
//using boost::math::policies::ignore_error;
//using boost::math::policies::throw_on_error;
//using boost::math::policies::user_error;


/*
namespace boost {
    namespace math {
        namespace policies {
            template <class T>
            T user_domain_error(const char* function, const char* message, const T& val)
            {
               cerr << "'" << function << "'" << endl;
               return std::numeric_limits<T>::quiet_NaN();
            }
        }
    }
}



typedef policy<
   domain_error<user_error>
> user_policy;
*/

using boost::math::erf_inv;


int main()
{
    double pvals[] = {-1.0, 1.0};

    cout << "  p       erf_inv(p)" << endl;
    for (const auto &p : pvals) {
        double x = -999.25;
        // double x = erf_inv(p, user_policy());
        try {
            x = erf_inv(p);
        } catch (const exception& e) {
            cerr << "*** Caught: " << e.what() << endl;
            cerr << "*** Type: " << typeid(e).name() << endl;
        }
        cout << scientific << setprecision(1) << p << " " << setprecision(17) << x << endl;
    }
}
