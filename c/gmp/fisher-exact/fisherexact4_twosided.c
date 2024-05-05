
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

/*
 *  Computer the Fisher exact test (two-sided) for a 2x2 contingency table
 *  given on the command line.
 *
 *  For example,
 *  $  ./fisherexact 10 15 12 8
 *  contingency table:
 *      10      15
 *      12       8
 *  support: a_min = 2,   a_max = 22  (len = 21)
 *  mode = 12
 *  two-sided p-value:
 *      num = 972322767000
 *      den = 4116715363800
 *      num/den = 41552255/175928007
 *      num/den = 0.2361889713216611
 */
 
int main(int argc, char *argv[])
{
    long int a, b, c, d;
    long int aa, bb, cc, dd;
    long int a_min, a_max;
    double mode;
    long int mode1, mode2;
    mpz_t anum;
    mpz_t num, numlo, numhi, den;
    mpz_t cnum1, cnum2;
    mpz_t less, greater, twosided;
    mpq_t t1, t2, mp;
    double p;

    if (argc < 5) {
        fprintf(stderr, "Use: fisherexact a b c d\n");
        exit(-1);
    }

    mpz_inits(anum, NULL);
    mpz_inits(num, numlo, numhi, den, NULL);
    mpz_inits(cnum1, cnum2, NULL);
    mpz_inits(twosided, NULL);

    a = strtol(argv[1], NULL, 10);
    b = strtol(argv[2], NULL, 10);
    c = strtol(argv[3], NULL, 10);
    d = strtol(argv[4], NULL, 10);

    printf("contingency table:\n");
    printf("%6ld  %6ld\n", a, b);
    printf("%6ld  %6ld\n", c, d);

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
    printf("support: a_min = %ld,   a_max = %ld  (len = %ld)\n",
           a_min, a_max, a_max - a_min + 1);

    mode = (a + c + 1.0)*(a + b + 1.0)/(a + b + c + d + 2.0);
    mode1 = (long int) ceil(mode) - 1;
    mode2 = (long int) floor(mode);
    if (mode1 != mode2) {
        printf("mode = {%ld, %ld}\n", mode1, mode2);    
    }
    else {
        printf("mode = %ld\n", mode1);
    }

    // mpz_bin_uiui(rop, n, k) computes the binomial coefficient
    // "n choose k" and stores it in rop.
    mpz_bin_uiui(cnum1, a + b, a);
    mpz_bin_uiui(cnum2, c + d, c);
    mpz_bin_uiui(den, a + b + c + d, a + c);
    mpz_mul(anum, cnum1, cnum2);

    // XXX Instead of implementing code branches for a < mode1 and
    // a > mode1, the columns could be swapped in the latter case.

    if ((a == mode1) || (a == mode2)) {
        // p-value is 1.
        mpz_set_ui(twosided, 1L);
        mpz_set_ui(den, 1L);
    }
    else if (a < mode1) {
        mpz_bin_uiui(cnum1, a + b, a_min);
        mpz_bin_uiui(cnum2, c + d, a + c - a_min);
        mpz_mul(numlo, cnum1, cnum2);
        for (aa = a_min; aa <= a; ++aa) {
            mpz_add(twosided, twosided, numlo);
            if (aa  < a) {
                mpz_mul_ui(numlo, numlo, b + a - aa);
                mpz_mul_ui(numlo, numlo, c + a - aa);
                mpz_divexact_ui(numlo, numlo, aa + 1);
                mpz_divexact_ui(numlo, numlo, d - (a - aa) + 1);
            }
        }
        mpz_bin_uiui(cnum1, a + b, a_max);
        mpz_bin_uiui(cnum2, c + d, a + c - a_max);
        mpz_mul(numhi, cnum1, cnum2);
        aa = a_max;
        while (mpz_cmp(numhi, numlo) <= 0) {
            // while (numhi <= numlo)
            mpz_add(twosided, twosided, numhi);
            mpz_mul_ui(numhi, numhi, aa);
            mpz_mul_ui(numhi, numhi, d - (a - aa));
            mpz_divexact_ui(numhi, numhi, b + (a - aa) + 1);
            mpz_divexact_ui(numhi, numhi, c + (a - aa) + 1);
            --aa;
        }
    }
    else {
        mpz_bin_uiui(cnum1, a + b, a_max);
        mpz_bin_uiui(cnum2, c + d, a + c - a_max);
        mpz_mul(numhi, cnum1, cnum2);
        for (aa = a_max; aa >= a; --aa) {
            mpz_add(twosided, twosided, numhi);
            if (aa > a) {
                mpz_mul_ui(numhi, numhi, aa);
                mpz_mul_ui(numhi, numhi, d - (a - aa));
                mpz_divexact_ui(numhi, numhi, b + (a - aa) + 1);
                mpz_divexact_ui(numhi, numhi, c + (a - aa) + 1);
            }
        }
        mpz_bin_uiui(cnum1, a + b, a_min);
        mpz_bin_uiui(cnum2, c + d, a + c - a_min);
        mpz_mul(numlo, cnum1, cnum2);
        aa = a_min;
        while (mpz_cmp(numhi, numlo) >= 0) {
            mpz_add(twosided, twosided, numlo);
            mpz_mul_ui(numlo, numlo, b + a - aa);
            mpz_mul_ui(numlo, numlo, c + a - aa);
            mpz_divexact_ui(numlo, numlo, aa + 1);
            mpz_divexact_ui(numlo, numlo, d - (a - aa) + 1);
            ++aa;
        }
    }

    printf("two-sided p-value:\n");
    printf("    num = ");
    mpz_out_str(stdout, 10, twosided);
    printf("\n");
    printf("    den = ");
    mpz_out_str(stdout, 10, den);
    printf("\n");

    mpq_inits(t1, t2, mp, NULL);
    mpq_set_z(t2, den);

    mpq_set_z(t1, twosided);
    mpq_div(mp, t1, t2);
    printf("    num/den = ");
    mpq_out_str(stdout, 10, mp);
    printf("\n");
    p = mpq_get_d(mp);
    printf("    num/den = %.16g\n", p);

    return 0;
}
