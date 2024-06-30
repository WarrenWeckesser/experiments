from math import comb, factorial
from typing import Literal, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from mpmath import mp

import mpsig


def _savgol_pinv_correction_factors(
    deriv: int,
    polyorder: int,
    x_center: float,
    x_scale: float,
    polyvander_column_scales: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    Computes the correction factors for the pseudoinversion for the Savitzky Golay
    coefficient computations. Please refer to the Notes section for more details.

    Parameters
    ----------
    deriv : int
        The derivative order and also row index of the pseudo-inverse matrix to be
        corrected. This corresponds to ``d`` in the Notes section.
    polyorder : int
        The polynomial order of the polynomial fit.
    polyvander_column_scales : np.ndarray of shape (polyorder + 1,)
        The Euclidean norms of the columns of the normalized polynomial Vandermonde matrix.
    x_center, x_scale : float
        The centering and scaling factors used to normalize the x-variable.
    delta : float
        The spacing between the grid points of the signal to be filtered.

    Returns
    -------
    pinv_correction_factors : np.ndarray of shape (polyorder + 1)
        The correction factors for the pseudo-inverse matrix row corresponding to
        ``pinv_row_idx``.
        All the elements that are not required are set to zero.

    Notes
    -----
    The correction takes into account that

    - the x-variable for the Vandermonde matrix was normalized as
        ``x_normalized = (x - x_center) / x_scale``,
    - the columns of the Vandermonde matrix ``J`` given by
        ``J = polyvander(x_normalized, polyorder)`` are scaled to have unit Euclidean norm
        by dividing by the column scales ``polyvander_column_scales``,
    - that the ``d``-th derivative order introduces a pre-factor of ``d! / delta**d``
        to the coefficients.

    While the first two steps ensure the numerical stability of the pseudoinverse
    computation, they distort the pseudoinverse of the polynomial fit. The correction
    factors are applied to the rows of the pseudo-inverse of ``J`` to recover the
    pseudoinverse of the polynomial fit based on the original columns of ``J``.

    The correction factors are computed as follows:

    ```
    phi_id = (d! / delta**d) * ((-1)**(i - d)) * comb(i, d) * ((1.0 / x_scale)**i) * 1.0 / col_scale[i] * (x_center ** (i - d))
    ```

    where

    - ``d`` is the derivative order and also the row index of the pseudo-inverse matrix
        to be corrected (the ``d``-th row corresponds to the ``d``-th derivative of the
        polynomial fit),
    - ``i`` is the iterator for the polynomial order,
    - ``comb(i, d)`` is the binomial coefficient,
    - ``col_scale[i]`` is the Euclidean norm of the ``i``-th column of the normalized
        polynomial Vandermonde matrix, and
    - ``delta`` is the spacing between the grid points of the signal to be filtered.

    These can be applied to correct the element in the ``d``-th row and the ``j``-th
    column of the distorted pseudo-inverse ``JD+`` as follows:

    ```
    J+[d, j] = sum(phi_id * JD+[i, j] for i in range(d, polyorder + 1))
    ```

    to obtain the corrected pseudo-inverse ``J+``.

    """  # noqa: E501

    # first, the signs are computed
    # incorporating the signs as -1 and 1 is not efficient, especially not when they
    # are computed by raising -1 to the power of the iterator
    # therefore, the signs are already pre-multiplied with the only constant factor in
    # the formula, namely ``deriv! * ((x_center * delta) ** (-deriv))``
    x_center_modified_for_deriv = factorial(deriv) * ((x_center * delta) ** (-deriv))
    # the following is equivalent to a multiplication of the factor with the signs,
    # however, since the signs are alternating, they are simply repeated as often as
    # necessary
    prefactors = [x_center_modified_for_deriv, -x_center_modified_for_deriv]
    n_full_repetitions, n_rest_repetitions = divmod(polyorder + 1 - deriv, 2)
    prefactors = np.array(
        prefactors * n_full_repetitions + prefactors[:n_rest_repetitions]
    )

    # then, the binomial coefficients are computed ...
    i_vect = np.arange(start=deriv, stop=polyorder + 1, dtype=np.int64)
    binomials = np.array([comb(i, deriv) for i in i_vect], dtype=np.int64)
    # ... followed by the x-factors
    x_factors = ((x_center / x_scale) ** i_vect) / polyvander_column_scales[deriv::]

    pinv_correction_factors = np.zeros(shape=(polyorder + 1,), dtype=np.float64)
    pinv_correction_factors[deriv::] = prefactors * binomials * x_factors
    return pinv_correction_factors


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

    pos_internal = pos
    if pos_internal is None:
        if rem == 0:
            pos_internal = halflen - 0.5
        else:
            pos_internal = halflen

    if not (0 <= pos_internal < window_length):
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
    x = np.arange(
        start=-pos_internal,
        stop=window_length - pos_internal,
        step=1.0,
        dtype=float,
    )

    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    # the stable version of the polynomial Vandermonde matrix is computed
    x_min, x_max = x.min(), x.max()
    x_center = 0.5 * (x_min + x_max)
    x_scale = 0.5 * (x_max - x_min)
    x_normalized = (x - x_center) / x_scale
    J_normalized_polyvander = np.polynomial.polynomial.polyvander(
        x_normalized, polyorder
    )

    # to stabilize the pseudo-inverse computation, the columns of the polynomial
    # Vandermonde matrix are normalized to have unit Euclidean norm
    j_column_scales = np.linalg.norm(J_normalized_polyvander, axis=0)
    J_normalized_polyvander /= j_column_scales[np.newaxis, ::]

    # then, the correction factors for the subsequent least squares problem are computed
    correction_factors = _savgol_pinv_correction_factors(
        deriv=deriv,
        polyorder=polyorder,
        polyvander_column_scales=j_column_scales,
        x_center=x_center,
        x_scale=x_scale,
        delta=delta,
    )

    # finally, the coefficients are obtained from solving the least squares problem
    # J.T @ coeffs = correction_factors
    return np.linalg.lstsq(
        J_normalized_polyvander.transpose(),
        correction_factors,
        rcond=None,
    )[0]


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
