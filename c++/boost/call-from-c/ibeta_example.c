#include <stdio.h>
#include "ibeta_wrap.h"

int main(int argc, char *argv[])
{
	double a, b, x, y;

	a = 0.5;
	b = 5e-7;
	x = 0.9999990000010001;
	y = boost_double_ibeta(a, b, x);
	printf("%g %g %g %g\n", a, b, x, y);

	return 0;
}
