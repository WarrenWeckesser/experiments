g++  -std=c++17 -I$BOOSTDIR bessel_demo.cpp -o bessel_demo
g++  -std=c++17 -I$BOOSTDIR -DBOOST_MATH_GIT=$(git -C $BOOSTDIR/libs/math rev-parse HEAD) cyl_bessel_i_demo.cpp -o cyl_bessel_i_demo
g++  -std=c++20 -I$BOOSTDIR  spherical_in_demo.cpp -o spherical_in_demo
