import numpy as np

# TO DO:
# * Simplify code in compute_k and unwrapi where possible.
# * Handle the "ambiguous case" like np.unwrap() (if that makes sense)
# * Decide how results outside the possible values of the input dtype should
#   be handled.  Error?  Or define the result to be the closest value among
#   the elements in the equivalence class that are also representable in the
#   dtype?
# * Review how np.unwrap handles the `discont` parameter, and implement in
#   unwrapi.

unsigned_type_map = {
    np.uint8: np.int8,
    np.uint16: np.int16,
    np.uint32: np.int32,
    np.uint64: np.int64,
    np.int8: np.int8,
    np.int16: np.int16,
    np.int32: np.int32,
    np.int64: np.int64
}


def compute_k(a, b, period):
    """
    Compute k such that ``b + k*period`` is the value from the
    equivalence class of `b` that is closest to `a`.

    `a` and `b` must have the same numpy integer type.

    `period` must be a positive Python integer, or a positive numpy integer
    of the same type as `a` and `b`. `period` must be at least 2.

    Returns a signed value, even when the inputs are unsigned.
    """
    qa, ra = np.divmod(a, period)
    qa = np.astype(qa, unsigned_type_map[qa.dtype.type])
    ra = np.astype(ra, unsigned_type_map[ra.dtype.type])
    qb, rb = np.divmod(b, period)
    qb = -np.astype(qb, unsigned_type_map[qb.dtype.type])
    rb = -np.astype(rb, unsigned_type_map[rb.dtype.type])
    qs, rs = np.divmod(ra + rb, period)
    qs = np.astype(qs, unsigned_type_map[qs.dtype.type])
    rs = np.astype(rs, unsigned_type_map[rs.dtype.type])
    if rs > period/2:
        qs += 1
    return qa + qb + qs


def unwrapi(x, period):
    # This function assumes:
    # * x is a 1-d ndarray with an integer dtype.
    # * period is either a Python integer or an NumPy integer
    #   with the same dtype as x.
    #
    # XXX/FIXME When period is even, this function does not yet handle
    # the ambiguous case the same as np.unwrap.  It can also generate a
    # warning when the period is even.
    period = abs(period)
    if period == 0:
        raise ValueError('period=0 is not implemented.')
    y = np.empty_like(x)
    if x.size == 0:
        return y
    if period == 1:
        y.fill(x[0])
        return y
    y[0] = x[0]
    for i, x1 in enumerate(x[1:]):
        k = compute_k(y[i], x1, period)
        if np.issubdtype(x.dtype, np.signedinteger):
            info = np.iinfo(x.dtype)
            if info.min/period <= k <= info.max/period:
                # period*k does not overflow the integer type.
                result = x1 + period*k
            else:
                result = x1
                b = info.max // period
                n, m = np.divmod(abs(k), b)
                if k > 0:
                    for _ in range(n):
                        result += b*period
                    result += m*period
                else:
                    for _ in range(n):
                        result -= b*period
                    result -= m*period
        else:
            if k < 0:
                result = x1 - period*(-k).astype(x1.dtype)
            else:
                result = x1 + period*(k).astype(x1.dtype)
        y[i+1] = result
    return y


def check_all_int8(period):
    skipped = []
    bad = []
    for i in range(-128, 128):
        for j in range(-128, 128):
            x = np.array([i, j], dtype=np.int8)
            ref = np.unwrap(x.astype(np.int16), period=period)
            if np.any(ref < -128) or np.any(ref > 127):
                skipped.append(x)
                continue
            y = unwrapi(x, period=period)
            if np.any(y != ref):
                bad.append(x)
    return skipped, bad


def check_all_uint8(period):
    skipped = []
    bad = []
    for i in range(0, 256):
        for j in range(0, 256):
            x = np.array([i, j], dtype=np.uint8)
            ref = np.unwrap(x.astype(np.int16), period=period)
            if np.any(ref < 0) or np.any(ref > 255):
                skipped.append(x)
                continue
            y = unwrapi(x, period=period)
            if np.any(y != ref):
                bad.append(x)
    return skipped, bad


def check_int16_random_sample(n, period, seed=121263137472525314065):
    rng = np.random.default_rng(seed)
    dt = np.dtype('int16')
    info = np.iinfo(dt)
    values = rng.integers(info.min, info.max+1, size=(n,2)).astype(dt)
    skipped = []
    bad = []
    for pair in values:
        ref = np.unwrap(pair.astype(np.int32), period=period)
        if np.any(ref < info.min) or np.any(ref > info.max):
            skipped.append(pair)
            continue
        y = unwrapi(pair, period=period)
        if np.any(y != ref):
            bad.append(pair)
    return skipped, bad


# XXX/FIXME: D.R.Y: make the dtype a parameter so this and the above function
# can be combined.
def check_uint16_random_sample(n, period, seed=121263137472525314065):
    rng = np.random.default_rng(seed)
    dt = np.dtype('uint16')
    info = np.iinfo(dt)
    values = rng.integers(info.min, info.max+1, size=(n,2)).astype(dt)
    skipped = []
    bad = []
    for pair in values:
        ref = np.unwrap(pair.astype(np.int32), period=period)
        if np.any(ref < info.min) or np.any(ref > info.max):
            skipped.append(pair)
            continue
        y = unwrapi(pair, period=period)
        if np.any(y != ref):
            bad.append(pair)
    return skipped, bad
