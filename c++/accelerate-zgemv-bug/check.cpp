#include <Accelerate/Accelerate.h>

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "npy.hpp"

int main()
{
    //
    // Read CC.npy
    //
    std::vector<unsigned long> cc_shape;
    bool cc_fortran;
    std::vector<std::complex<double>> CC;

    npy::LoadArrayFromNumpy(
        "CC.npy",
        cc_shape,
        cc_fortran,
        CC
    );
    if (cc_shape.size() != 2)
        throw std::runtime_error("CC.npy is not 2-dimensional");
    if (cc_fortran)
        throw std::runtime_error("CC.npy must be C-order (row-major)");

    const int M = static_cast<int>(cc_shape[0]);
    const int N = static_cast<int>(cc_shape[1]);

    std::cout << "CC has shape (" << M << ", " << N << ")" << std::endl;

    //
    // Read weights.npy
    //
    std::vector<unsigned long> w_shape;
    bool w_fortran;
    std::vector<std::complex<double>> weights;

    npy::LoadArrayFromNumpy(
        "weights.npy",
        w_shape,
        w_fortran,
        weights
    );
    if (w_shape.size() != 1)
        throw std::runtime_error("weights.npy is not a vector");
    if (static_cast<int>(w_shape[0]) != N)
        throw std::runtime_error("length of weights does not equal the second dimension of CC");

    //
    // y = CC @ weights
    //
    std::vector<std::complex<double>> y(M);

    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);

    cblas_zgemv(
        CblasRowMajor,
        CblasNoTrans,
        M,
        N,
        &alpha,
        CC.data(),
        N,
        weights.data(),
        1,
        &beta,
        y.data(),
        1
    );

    //
    // Print y[17]
    //
    std::cout << "@ is matrix multiplication computed with cblas_zgemv.\n" << std::endl;
    std::cout << "(CC @ weights)[17] = " << y[17] << std::endl;

    //
    // y17 = CC[17,:] @ weights
    //
    const int row = 17;
    std::vector<std::complex<double>> y17(1);

    cblas_zgemv(
        CblasRowMajor,
        CblasNoTrans,
        1,
        N,
        &alpha,
        CC.data() + row * N,
        N,
        weights.data(),
        1,
        &beta,
        y17.data(),
        1
    );
    
    std::cout << "CC[17,:] @ weights = " << y17[0] << std::endl;
}
