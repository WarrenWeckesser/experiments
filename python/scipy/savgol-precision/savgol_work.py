from math import comb, fsum
from typing import Literal, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from mpmath import mp
from scipy._lib._util import float_factorial

import mpsig


def _pinv_correction_factors_normalized_polyvander(
    pinv_row_idx: int,
    polyorder: int,
    polyvander_column_scales: np.ndarray,
    x_center: float,
    x_scale: float,
) -> np.ndarray:
    """
    Computes the correction factors for the pseudo-inverse of a normalized polynomial
    Vandermonde matrix where the x-variable was normalized as ``x_scaled = (x - x_center) / x_scale``.
    After this normalization, the columns of the Vandermonde matrix ``A`` given by
    ``A = polyvander(x_scaled, polyorder)`` are scaled to have unit Euclidean norm.
    These steps ensure the numerical stability of the pseudo-inverse computation, but
    distorts the pseudo-inverse of the polynomial fit. It will then be matched to the
    scaled columns of ``A`` who were themselves normalized before, rather than the
    original columns of ``A`` that are based on ``x``.

    This function computes the correction factors to be applied to the rows of the
    pseudo-inverse of ``A`` to recover the pseudo-inverse of the polynomial fit based
    on the original columns of ``A``.
    Multiplying out all scaling and centering factors and rearranging terms gives the
    following equation for the correction factors:

    ```
    phi_ik = ((-1)**(i - k)) * comb(i, k) * ((1.0 / x_scale)**i) * 1.0 / col_scale[i] * (x_center ** (i - k))
    ```

    where

    - ``k`` is the row index of the pseudo-inverse matrix to be corrected (the ``k``-th
        row corresponds to the ``k``-th coefficient of the polynomial fit, e.g., ``k=0``
        corresponds to the constant term ``a0``),
    - ``i`` is the iterator for the polynomial order
    - ``comb(i, k)`` is the binomial coefficient, and
    - ``col_scale[i]`` is the Euclidean norm of the ``i``-th column of the normalized
        polynomial Vandermonde matrix

    These can be applied to correct the element in the ``k``-th row and the ``j``-th
    column of the distorted pseudo-inverse ``JD+`` as follows:

    ```
    J+[k, j] = sum(phi_ik * JD+[i, j] for i in range(k, polyorder + 1))
    ```

    to obtain the corrected pseudo-inverse ``J+``.

    Parameters
    ----------
    pinv_row_idx : int
        The row index of the pseudo-inverse matrix to be corrected. This corresponds to
        ``k`` in the above equation.
    polyorder : int
        The polynomial order of the polynomial fit.
    polyvander_column_scales : np.ndarray of shape (polyorder + 1,)
        The Euclidean norms of the columns of the normalized polynomial Vandermonde matrix.
    x_center, x_scale : float
        The centering and scaling factors used to normalize the x-variable.

    Returns
    -------
    pinv_correction_factors : np.ndarray of shape (polyorder + 1 - pinv_row_idx,)
        The correction factors for the pseudo-inverse matrix row corresponding to
        ``pinv_row_idx``.

    """  # noqa: E501

    # first, the signs are determined by the (-1)**(i - k) factor ...
    i_vect = np.arange(start=pinv_row_idx, stop=polyorder + 1, dtype=np.int64)
    signs = (-1) ** (i_vect - pinv_row_idx)
    # ... followed by the binomial coefficients ...
    binomials = np.array([comb(i, pinv_row_idx) for i in i_vect], dtype=np.int64)
    # ... then the factors for the x-variable
    x_factors = (
        (x_center ** (i_vect - pinv_row_idx))
        / (x_scale**i_vect)
        / polyvander_column_scales[pinv_row_idx::]
    )

    return signs * binomials * x_factors


def _correct_pinv_row_normalized_polyvander(
    pinv_row_idx: int,
    pinv_scaled_polyvander: np.ndarray,
    polyorder: int,
    polyvander_column_scales: np.ndarray,
    x_center: float,
    x_scale: float,
) -> np.ndarray:
    """
    Corrects the ``k``-th row of pseudo-inverse of a normalized polynomial Vandermonde
    matrix to obtain the ``k``-th row of the pseudo-inverse of the polynomial fit based
    on the original columns of the Vandermonde matrix.
    For the correction it relies on floating point accurate summation to avoid loss of
    precision due to cancellation errors with high polynomial orders.
    Please refer to the documentation of the function :func:`_pinv_correction_factors_normalized_polyvander`
    for a detailed explanation of the correction factors.

    Parameters
    ----------
    pinv_row_idx : int
        The row index of the pseudo-inverse matrix to be corrected. This corresponds to
        ``k`` in the above equation.
    pinv_scaled_polyvander : np.ndarray of shape (polyorder + 1, m)
        The pseudo-inverse of the normalized polynomial Vandermonde matrix.
    polyorder : int
        The polynomial order of the polynomial fit.
    polyvander_column_scales : np.ndarray of shape (polyorder + 1,)
        The Euclidean norms of the columns of the normalized polynomial Vandermonde matrix.
    x_center, x_scale : float
        The centering and scaling factors used to normalize the x-variable.

    Returns
    -------
    pinv_corrected_polyvander : np.ndarray of shape (polyorder + 1, m)
        The corrected pseudo-inverse of the polynomial fit that corresponds to the
        original columns of the polynomial Vandermonde matrix.

    """  # noqa: E501

    correction_factors = _pinv_correction_factors_normalized_polyvander(
        pinv_row_idx=pinv_row_idx,
        polyorder=polyorder,
        polyvander_column_scales=polyvander_column_scales,
        x_center=x_center,
        x_scale=x_scale,
    )
    return np.array(
        [
            fsum(pinv_scaled_polyvander[pinv_row_idx::, j] * correction_factors)
            for j in range(pinv_scaled_polyvander.shape[1])
        ]
    )


def super_stabilised_savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: Union[float, int] = 1.0,
    pos: Optional[int] = None,
    use: Literal["conv", "dot"] = "conv",
) -> np.ndarray:

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than " "window_length.")

    if use not in ["conv", "dot"]:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # Form the design matrix A. The columns of A are powers of the integers
    # from -pos to window_length - pos - 1. The powers (i.e., rows) range
    # from 0 to polyorder. (That is, A is a vandermonde matrix, but not
    # necessarily square.)
    x = np.arange(-pos, window_length - pos, dtype=float)

    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    # the stable version of the polynomial Vandermonde matrix is computed
    x_min, x_max = x.min(), x.max()
    x_center = 0.5 * (x_min + x_max)
    x_scale = 0.5 * (x_max - x_min)
    x_normalized = (x - x_center) / x_scale
    A_scaled_polyvander = np.polynomial.polynomial.polyvander(x_normalized, polyorder)

    # to stabilize the pseudo-inverse computation, the columns of the polynomial
    # Vandermonde matrix are normalized to have unit Euclidean norm
    a_column_scales = np.linalg.norm(A_scaled_polyvander, axis=0)
    A_scaled_polyvander /= a_column_scales[np.newaxis, ::]

    # then, the pseudo-inverse is computed and corrected to obtain the pseudo-inverse
    # of the polynomial fit based on the columns the original polynomial Vandermonde
    # matrix based on the original x-values
    A_pinv_distorted = np.linalg.pinv(A_scaled_polyvander)

    return (
        float_factorial(deriv) / delta**deriv
    ) * _correct_pinv_row_normalized_polyvander(
        pinv_row_idx=deriv,
        pinv_scaled_polyvander=A_pinv_distorted,
        polyorder=polyorder,
        polyvander_column_scales=a_column_scales,
        x_center=x_center,
        x_scale=x_scale,
    )


def stabilised_savgol_coeffs(
    window_length,
    polyorder,
    deriv=0,
    delta=1.0,
    pos=None,
    use="conv",
):
    # An alternative method for finding the coefficients when deriv=0 is
    #    t = np.arange(window_length)
    #    unit = (t == pos).astype(int)
    #    coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
    # The method implemented here is faster.

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than " "window_length.")

    if use not in ["conv", "dot"]:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # Form the design matrix A. The columns of A are powers of the integers
    # from -pos to window_length - pos - 1. The powers (i.e., rows) range
    # from 0 to polyorder. (That is, A is a vandermonde matrix, but not
    # necessarily square.)
    x = np.arange(-pos, window_length - pos, dtype=float)
    x_abs_max = max(pos, window_length - pos - 1)
    # The following avoids zero division in case `polyorder=0` and
    # `window_length=1`.
    scale = max(x_abs_max, 1.0)
    x /= scale
    delta *= scale

    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x**order

    # y determines which order derivative is returned.
    y = np.zeros(polyorder + 1)
    # The coefficient assigned to y[deriv] scales the result to take into
    # account the order of the derivative and the sample spacing.
    y[deriv] = float_factorial(deriv) / (delta**deriv)
    # Find the least-squares solution of A*c = y
    # coeffs, _, _, _ = lstsq(A, y, rcond=None)  # Specify rcond with numpy
    print(f'{np.linalg.cond(A)}; shape is {A.shape}')
    coeffs, _, _, _ = lstsq(A, y)
    return coeffs


def simple_savgol_coeffs(window_length, polyorder, pos=None):
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd when pos is None")
    if pos is None:
        pos = window_length // 2
    t = np.arange(window_length)
    unit = (t == pos).astype(int)
    p = np.polynomial.Polynomial.fit(t, unit, deg=polyorder)
    coeffs = p(t)
    return coeffs[::-1]


def check_savgol_coeffs(c, cref):
    relerr = np.array(
        [
            float(abs((c0 - cref0) / cref0)) if cref != 0 else np.inf
            for c0, cref0 in zip(c, cref)
        ]
    )
    return relerr


if __name__ == "__main__":
    mp.dps = 150

    window_len = 51
    order = 10
    pos = 35
    deriv = 5
    delta = 0.5

    coeffs = mpsig.savgol_coeffs(window_len, order, pos=pos, deriv=deriv, delta=delta)
    if deriv == 0:
        cnp = simple_savgol_coeffs(window_len, order, pos=pos)

    c_sup_s = super_stabilised_savgol_coeffs(
        window_len, order, pos=pos, deriv=deriv, delta=delta
    )
    # c_s = stabilised_savgol_coeffs(window_len, order, pos=pos, deriv=deriv, delta=delta)

    if deriv == 0:
        e_cnp = check_savgol_coeffs(cnp, coeffs)
    e_c_sup_s = check_savgol_coeffs(c_sup_s, coeffs)
    # e_c_s = check_savgol_coeffs(c_s, coeffs)

    if deriv == 0:
        plt.plot(
            e_cnp,
            "8",
            alpha=0.65,
            label=f"numpy Polynomial.fit\n(rel err: max {e_cnp.max():8.2e}, median {np.median(e_cnp):8.2e})",
        )

    plt.plot(
        e_c_sup_s,
        "d",
        alpha=0.65,
        label=f"super stabilised_savgol_coeffs\n(rel err: max {e_c_sup_s.max():8.2e}, median {np.median(e_c_sup_s):8.2e}",
    )
    # plt.plot(e_c_s, 'x', alpha=0.65,
    #             label=f'stabilised_savgol_coeffs\n(rel err: max {e_c_s.max():8.2e}, median {np.median(e_c_s):8.2e}')
    plt.legend(shadow=True, framealpha=1)
    plt.grid()
    plt.title(
        f"Relative Error of Savitzky-Golay Coefficients\n{window_len=}  {order=}  {pos=}, {deriv=}"
    )
    plt.xlabel("Coefficient index")
    plt.ylabel("Relative error of coefficient")

    fig, ax = plt.subplots()

    ax.plot(coeffs, "o", alpha=0.65, label="mpsig.savgol_coeffs")
    if deriv == 0:
        ax.plot(cnp, "8", alpha=0.65, label="numpy Polynomial.fit")

    ax.plot(c_sup_s, "d", alpha=0.65, label="super stabilised_savgol_coeffs")
    # ax.plot(c_s, 'x', alpha=0.65, label='stabilised_savgol_coeffs')

    ax.legend(shadow=True, framealpha=1)
    ax.grid()
    ax.set_title(f"Savitzky-Golay Coefficients\n{window_len=}  {order=}  {pos=}")
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Coefficient value")

    plt.show()
