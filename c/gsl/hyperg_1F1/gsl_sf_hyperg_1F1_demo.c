
#include <stdio.h>
#include <gsl/gsl_sf_hyperg.h>

int main(int argc, char *argv[])
{
    for (int n = -1; n > -5; --n) {
        printf("%3d ", n);
        for (double x = 1; x < 3; x += 1.0) {
            double y1 = gsl_sf_hyperg_1F1_int(n, n, x);
            double y2 = gsl_sf_hyperg_1F1(n, n, x);
            printf("%20.15f %20.15f", y1, y2);
        }
        printf("\n");
    }

    return 0;
}
