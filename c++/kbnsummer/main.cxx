

#include <cstdio>
#include <complex>
#include "kbnsummer.hh"

int main(void)
{
    double data[] = {1.0, 1e-16, -0.5, -0.125, 1e-16, -0.125, -0.125, 1.0, -1.125};
    KBNSummer<double> total;

    for (int i = 0; i < sizeof(data)/sizeof(data[0]); ++i) {
        total += data[i];
    }
    printf("%24.17e  %24.17e %24.17e\n\n",
           total.sum(), total.c(), static_cast<double>(total));
    double p[] = {
        -0.41253261766461263,
        41287272281118.43,
        -1.4727977348624173e-14,
        5670.3302557520055,
        2.119245229045646e-11,
        -0.003679264134906428,
        -6.892634568678797e-14,
        -0.0006984744181630712,
        -4054136.048352595,
        -1003.101760720037,
        -1.4436349910427172e-17,
        -41287268231649.57
    };
    KBNSummer<double> ptotal;

    for (int i = 0; i < sizeof(p)/sizeof(p[0]); ++i) {
        ptotal += p[i];
    }
    printf("p:  acc=%24.17e\n      c=%24.17e\n    sum=%24.17e\n",
           ptotal.acc(), ptotal.c(), static_cast<double>(ptotal));
    printf("\n");

    std::complex<double> zp[] = {
        {-0.41253261766461263,     1.0},
        {41287272281118.43,        0.0},
        {-1.4727977348624173e-14,  1.0},
        {5670.3302557520055,       0.0},
        {2.119245229045646e-11,    0.0},
        {-0.003679264134906428,    1e-9},
        {-6.892634568678797e-14,   0.0},
        {-0.0006984744181630712,  -1.5},
        {-4054136.048352595,       0.0},
        {-1003.101760720037,       0.5},
        {-1.4436349910427172e-17, -1.0},
        {-41287268231649.57,       0.0}
    };
    KBNSummer<double> zptotal_real, zptotal_imag;
    KBNSummer<std::complex<double>> zptotal;

    for (int i = 0; i < sizeof(zp)/sizeof(zp[0]); ++i) {
        zptotal += zp[i];
        zptotal_real += zp[i].real();
        zptotal_imag += zp[i].imag();
    }
    std::complex<double> zptot = zptotal;
    printf("zp: acc=(%24.17e, %24.17e)\n      c=(%24.17e, %24.17e)\n    sum=(%24.17e, %24.17e)\n",
           zptotal.acc().real(), zptotal.acc().imag(),
           zptotal.c().real(), zptotal.c().imag(),
           zptot.real(), zptot.imag());

    printf("\n");
    printf("%24.17e  %24.17e %24.17e\n", zptotal_real.acc(), zptotal_real.c(), static_cast<double>(zptotal_real));
    printf("%24.17e  %24.17e %24.17e\n", zptotal_imag.acc(), zptotal_imag.c(), static_cast<double>(zptotal_imag));

}
