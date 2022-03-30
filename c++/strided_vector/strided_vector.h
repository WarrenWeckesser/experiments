#include <cstddef>

template<typename T>
class StridedVector
{
private:
    T *base;
    const size_t len;
    const ptrdiff_t stride;

public:

    StridedVector(T *base, size_t len, ptrdiff_t stride) : base(base), len(len), stride(stride) {}

    ptrdiff_t
    get_stride()
    {
        return stride;
    }

    size_t
    size()
    {
        return len;
    }

    T&
    operator[](size_t k)
    {
        return *(T *) ((char *) base + k*stride);
    }
};


// Helper function for template type deduction.
// (This could probably be eliminated when using C++17.)

template<typename T>
StridedVector<T>
make_strided_vector(T *base, size_t len, ptrdiff_t stride)
{
    return StridedVector<T>(base, len, stride);
}
