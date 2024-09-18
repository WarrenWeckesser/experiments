#ifndef SIMPLE_ARRAY_H
#define SIMPLE_ARRAY_H

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <stdexcept>
#include <vector>

typedef std::vector<size_t> multiindex_t;
typedef std::vector<size_t> shape_t;

//
// Note that this returns 1 if shape is {}.
// That is intentional.
//
static size_t size_of_shape(shape_t shape)
{
    size_t product = 1;
    // XXX this doesn't check for overflow of product.
    for (auto &d: shape) {
        product *= d;
    }
    return product;
}

static std::vector<size_t> default_strides(shape_t shape)
{
    size_t ndim = shape.size();
    shape_t strides(ndim);
    size_t stride = 1;
    for (int k = ndim; k > 0; --k) {
        strides[k - 1] = stride;
        stride *= shape[k - 1];
    }
    return strides;
}

class MultiIndexIterator 
{
    using iterator_category = std::input_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = multiindex_t;
    using pointer           = multiindex_t*;
    using reference         = const multiindex_t&;

    shape_t _shape;
    multiindex_t index;
    bool done;

public:

    MultiIndexIterator(const shape_t shape) : _shape(shape)
    {
        done = false;
        index = multiindex_t(shape.size());
    }

    MultiIndexIterator(const shape_t shape, bool done) : _shape(shape), done(done)
    {
        index = multiindex_t(shape.size());
    }

    reference operator*() const{
        return index;
    }

    MultiIndexIterator& operator++() {
        if (done) {
            return *this;
        }
        done = true;
        for (size_t k = index.size(); k > 0; --k) {
            size_t i = index.at(k-1) + 1;
            if (i < _shape.at(k-1)) {
                index.at(k-1) = i;
                done = false;
                break;
            }
            else {
                index.at(k-1) = 0;
            }
        }
        return *this;
    }  

    friend bool operator==(const MultiIndexIterator& a,
                           const MultiIndexIterator& b)
    {
        size_t n = a.index.size();
        if (n != b.index.size()) {
            return false;
        }
        if (a.done != b.done) {
            return false;
        }
        for (size_t k = 0; k < n; ++k) {
            if (a.index[k] != b.index[k]) {
                return false;
            }
        }
        return true;
    };

    friend bool operator!=(const MultiIndexIterator& a,
                           const MultiIndexIterator& b) {
        size_t n = a.index.size();
        if (n != b.index.size()) {
            return true;
        }
        if (a.done != b.done) {
            return true;
        }
        for (size_t k = 0; k < n; ++k) {
            if (a.index[k] != b.index[k]) {
                return true;
            }
        }
        return false;
    };

    MultiIndexIterator begin() {
        return MultiIndexIterator(_shape);
    }

    MultiIndexIterator end() {
        multiindex_t end(_shape.size());
        return MultiIndexIterator(_shape, true);
    }

};

template<typename T>
class SimpleArray {

public:

    // The shape of the array.
    shape_t _shape;

    // The array data stored serialized as a 1-d vector.
    std::vector<T> _data;

    // _index_strides is derived from _shape; it is stored for convenience.
    // By default, it is the reverse cumulative product of reversed _shape,
    // starting from 1.
    // E.g. if _shape = {7, 2, 3, 5}, then _index_strides is {30, 15, 5, 1}.
    shape_t _index_strides;

    // FIXME: Eliminate repeated code in the two constructors.

    SimpleArray(shape_t shape, std::vector<T> data, shape_t index_strides):
                _shape(shape), _data(data), _index_strides(index_strides)
    {
        size_t size = size_of_shape(_shape);
        if (size != _data.size()) {
            std::ostringstream os;
            os << "product of dimensions in `shape` (" << size << ") "
               << "does not equal size of `data` (" << _data.size() << ")" << std::endl;
            throw std::invalid_argument(os.str());
        }
    }

    SimpleArray(shape_t shape, std::vector<T> data): _shape(shape), _data(data)
    {
        size_t size = size_of_shape(_shape);
        if (size != _data.size()) {
            std::ostringstream os;
            os << "product of dimensions in `shape` (" << size << ") "
               << "does not equal size of `data` (" << _data.size() << ")" << std::endl;
            throw std::invalid_argument(os.str());
        }
        _index_strides = default_strides(_shape);
    }

    T flat(size_t index) {
        return _data.at(index);
    }

    size_t flat_size() {
        return _data.size();
    }

    shape_t shape() {
        return _shape;
    }

    T operator[](multiindex_t index) {
        if (index.size() != _shape.size()) {
            throw std::invalid_argument("index must have the same size as the shape");
        }
        size_t flat_index = 0;
        for (size_t k = 0; k < index.size(); ++k) {
            if (index.at(k) < 0 || index.at(k) >= _shape.at(k)) {
                throw std::invalid_argument("index out of bounds");
            }
            flat_index += _index_strides.at(k)*index[k];
        }
        return _data.at(flat_index);
    }

    MultiIndexIterator index_iterator() {
        return MultiIndexIterator(_shape);
    }

    void dump(const std::string name) const
    {
        std::cerr << "array dump: name='" << name << "'" << std::endl;
        std::cerr << "_shape: ";
        for (auto &d: _shape) {
            std::cerr << " " << d; 
        }
        std::cerr << std::endl;
        std::cerr << "_index_strides: ";
        for (auto &d: _index_strides) {
            std::cerr << " " << d; 
        }
        std::cerr << std::endl;
        std::cerr << "data.size(): " << _data.size() << std::endl;
    }
};

template<typename T>
SimpleArray<T> make_scalar_array(T item)
{
    return SimpleArray{shape_t{}, std::vector<T>{item}};
}

template<typename T>
SimpleArray<T> make_1d_array(std::vector<T> values)
{
    return SimpleArray{shape_t{values.size()}, values};
}

template<typename T>
shape_t broadcast_shape(std::vector<SimpleArray<T>> arrays)
{
    shape_t broadcast_shape{};
    for (auto &a: arrays) {
        shape_t shape = a.shape();
        while (shape.size() < broadcast_shape.size()) {
            shape.insert(shape.begin(), 1);
        }
        while (broadcast_shape.size() < shape.size()) {
            broadcast_shape.insert(broadcast_shape.begin(), 1);
        }
        for (size_t k = 0; k < broadcast_shape.size(); ++k) {
            if (broadcast_shape[k] != 1) {
                if (shape[k] != 1 && broadcast_shape[k] != shape[k]) {
                    throw std::runtime_error("not compatible for broadcasting");
                }
            }
            else {
                broadcast_shape[k] = shape[k];
            }
        }
    }
    return broadcast_shape;
}

//
// This function assumes that `a` can, in fact, be broadcast to `shape`.
// It does not validate the input parameters.
//
template<typename T>
SimpleArray<T> broadcast_to(SimpleArray<T> a, shape_t shape)
{
    size_t a_ndim = a._shape.size();
    size_t b_ndim = shape.size();
    SimpleArray b = a;
    b._shape = shape;
    b._index_strides = shape_t(b_ndim);
    for (size_t k = 0; k < b_ndim; ++k) {
        if (k < a_ndim) {
            if (a._shape[a_ndim - k - 1] == 1 && shape[b_ndim - k - 1] != 1) {
                b._index_strides[b_ndim - k - 1] = 0;
            }
            else {
                b._index_strides[b_ndim - k - 1] = a._index_strides[a_ndim - k - 1];
            }
        }
        else {
            b._index_strides[b_ndim - k - 1] = 0;
        }
    }
    return b;
}

#endif
