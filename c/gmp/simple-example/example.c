#include <stdio.h>
#include <gmp.h>

int main()
{
    mpf_t x;
    mpf_t y;
    mpf_t z;

    mpf_set_default_prec(256);

    mpf_inits(x, y, z, NULL);

    mpf_set_d(x, 1.0);
    mpf_set_d(y, 3.0);
    mpf_div(z, x, y);
    printf("z = ");
    mpf_out_str(stdout, 10, 0, z);
    printf("\n");

    mpf_clears(x, y, z, NULL);

    return 0;
}
