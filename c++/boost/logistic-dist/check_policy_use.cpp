#include <iostream>
#include <boost/math/special_functions/gamma.hpp>

using namespace boost::math::policies;
using boost::math::tgamma;

// Define a new policy *not* internally promoting RealType to double:
typedef policy<promote_double<false>> my_policy;

int main()
{
      double x = 87.5;

      // Call the function, applying the new policy:
      double t1 = tgamma(x, my_policy());

      // Alternatively we could use helper function make_policy,
      // and concisely define everything at the call site:
      double t2 = tgamma(x, make_policy(promote_double<false>()));

      double t3 = tgamma(x);

      std::cout << std::setprecision(17) << "t1: " << t1 << std::endl;
      std::cout << std::setprecision(17) << "t2: " << t2 << std::endl;
      std::cout << std::setprecision(17) << "t3: " << t3 << std::endl;
}