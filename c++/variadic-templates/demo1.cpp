//
// A simple example of a variadic template.
//

#include <cstdio>

template<typename T>
T sum(T x)
{
    return x;
}

template<typename T, typename... Args>
T sum(T x, Args... args)
{
    return x + sum(args...);
}


int main()
{
    double t = sum(1.0, 2.0, 3.0, 4.0);
    printf("t = %lf\n", t);
    double t1 = sum(7.5);
    printf("t1 = %lf\n", t1);
}
