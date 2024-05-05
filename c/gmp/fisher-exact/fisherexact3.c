
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

/*
 *  Computer the Fisher exact test for a 2x2 contingency table
 *  given on the command line.
 *
 *  For example,
 *  $  ./fisherexact 10 15 12 8
 *      10      15
 *      12       8
 *  less:      0.150710639836
 *  greater:   0.949312232967
 *  two-sided: 0.236188971322
 *
 *  This is a "brute force" calculation.  It will take a very long
 *  time if the array elements are large.
 *
 */
 
int main(int argc, char *argv[])
{
    long int a, b, c, d;
    long int aa, bb, cc, dd;
    long int a_min, a_max;
    mpz_t anum;
    mpz_t num, den;
    mpz_t cnum1, cnum2;
    mpz_t less, greater, twosided;
    mpq_t t1, t2, mp;
    double p;

    if (argc < 5) {
        fprintf(stderr, "Use: fisherexact a b c d\n");
        exit(-1);
    }

    mpz_inits(anum, NULL);
    mpz_inits(num, den, NULL);
    mpz_inits(cnum1, cnum2, NULL);
    mpz_inits(less, greater, twosided, NULL);

    a = strtol(argv[1], NULL, 10);
    b = strtol(argv[2], NULL, 10);
    c = strtol(argv[3], NULL, 10);
    d = strtol(argv[4], NULL, 10);

    if ((a + b) < (a + c)) {
        a_max = a + b;
    }
    else {
        a_max = a + c;
    }
    a_min = a - d;
    if (a_min < 0) {
        a_min = 0;
    }

    // mpz_bin_uiui(rop, n, k) computes the binomial coefficient
    // "n choose k" and stores it in rop.
    mpz_bin_uiui(cnum1, a + b, a);
    mpz_bin_uiui(cnum2, c + d, c);
    mpz_bin_uiui(den, a + b + c + d, a + c);
    mpz_mul(anum, cnum1, cnum2);

    printf("Iterating over %ld matrices.\n", a_max - a_min + 1);

    mpz_bin_uiui(cnum1, a + b, a_min);
    mpz_bin_uiui(cnum2, c + d, a + c - a_min);
    mpz_mul(num, cnum1, cnum2);

    printf("a_min = %ld\n", a_min);
    for (aa = a_min; aa <= a_max; ++aa) {
        //mpz_out_str(stdout, 10, num);
        //printf("\n");
        if (aa <= a) {
            mpz_add(less, less, num);
        }
        if (aa >= a) {
            mpz_add(greater, greater, num);
        }
        if (mpz_cmp(anum, num) >= 0) {
            mpz_add(twosided, twosided, num);
        }
        mpz_mul_ui(num, num, b + a - aa);
        mpz_mul_ui(num, num, c + a - aa);
        mpz_divexact_ui(num, num, aa + 1);
        mpz_divexact_ui(num, num, d - (a - aa) + 1);
    }

    printf("%6ld  %6ld\n", a, b);
    printf("%6ld  %6ld\n", c, d);

    mpq_inits(t1, t2, mp, NULL);
    mpq_set_z(t2, den);

    mpq_set_z(t1, less);
    mpq_div(mp, t1, t2);
    p = mpq_get_d(mp);
    printf("less:      %.12g\n", p);

    mpq_set_z(t1, greater);
    mpq_div(mp, t1, t2);
    p = mpq_get_d(mp);
    printf("greater:   %.12g\n", p);

    mpq_set_z(t1, twosided);
    mpq_div(mp, t1, t2);
    p = mpq_get_d(mp);
    printf("two-sided: %.12g\n", p);

    return 0;
}
