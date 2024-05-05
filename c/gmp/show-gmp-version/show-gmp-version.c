
#include <stdio.h>
#include <gmp.h>

int main(int argc, char *argv[])
{
	printf("GMP version is %d.%d.%d\n", __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR,  __GNU_MP_VERSION_PATCHLEVEL);
	return 0;
}
