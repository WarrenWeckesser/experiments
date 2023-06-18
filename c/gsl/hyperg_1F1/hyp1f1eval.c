
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_hyperg.h>

int main(int argc, char *argv[])
{
    gsl_sf_result result;

    if (argc != 4) {
        printf("use: %s a b x\n", argv[0]);
        exit(-1);
    }
    double a = strtod(argv[1], NULL);
    double b = strtod(argv[2], NULL);
    double x = strtod(argv[3], NULL);
    printf("a = %24.16f\n", a);
    printf("b = %24.16f\n", b);
    printf("x = %24.16f\n", x);
    // The return value of gsl_set_error_handler_off() is ignored.
    gsl_set_error_handler_off();
    int status = gsl_sf_hyperg_1F1_e(a, b, x, &result);
    printf("status = %d\n", status);
    if (status == 0) {
        printf("y = %.17e\n", result.val);
        printf("abs err est = %.9e\n", result.err);
    }
    else {
        printf("error: %s\n", gsl_strerror(status));
    }

    return 0;
}
