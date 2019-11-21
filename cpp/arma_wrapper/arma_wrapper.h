#ifndef ARMA_WRAPPER_H
#define ARMA_WRAPPER_H

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <armadillo>
#include <Python/Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/npy_common.h>
#include <pybind11/stl.h>

namespace std
{

template <typename Num>
constexpr auto cbegin(const arma::Mat<Num> &matrix) -> decltype(std::begin(matrix))
{

    return std::begin(matrix);
}

template <typename Num>
constexpr auto cend(const arma::Mat<Num> &matrix) -> decltype(std::end(matrix))
{

    return std::end(matrix);
}

} // namespace std

namespace aw
{

namespace py = pybind11;

using npulong = npy_ulong;
using arma::Col;
using arma::Cube;
using arma::Mat;
template <typename T>
using py_arr = py::array_t<T, py::array::f_style | py::array::forcecast>;
template <typename T>
using py_arr_c = py::array_t<T, py::array::c_style | py::array::forcecast>;
using npint = npy_int;
using npdouble = npy_double;
using intcube = arma::Cube<npint>;
using intmat = arma::Mat<npint>;
using dcube = arma::Cube<npdouble>;
using dmat = arma::Mat<npdouble>;
using iuvec = arma::Col<npulong>;
using std::begin;
using std::cbegin;
using std::cend;
using std::end;

/**
 * These functions simply converts a carray of data into the a matrix with the given shape. It does not ensure that
 * the given shape is the correct number of elements. It does not copy the data if fortran_order is false. It assumes
 * that the original matrix is in c-major order.  
 */
template <template <typename> typename T1, typename num>
std::enable_if_t<std::is_same<T1<num>, Cube<num>>::value, T1<num>>
conv_to(const num *numbers, const std::vector<npulong> &shape, bool fortran_order = true)
{

    Cube<num> readin_mat(numbers, shape.at(2), shape.at(1), shape.at(0));
    Cube<num> return_mat(shape.at(1), shape.at(2), shape.at(0));

    if (fortran_order)
    {
        for (npulong i = 0; i < return_mat.n_slices; ++i)
        {
            return_mat.slice(i) = readin_mat.slice(i).t();
        }
    }
    return return_mat;
}

template <typename num, npulong dim, typename InfoType>
std::vector<npulong> extract_shape(const InfoType &info)
{

    std::vector<npulong> shape(dim, 1);
    std::string format_err_msg = "The format descriptor strings are not the same. Are you using the right template"
                                 " specialization?";

    if (info.format != py::format_descriptor<num>::value)
    {
        throw std::runtime_error(format_err_msg);
    }

    if (sizeof(num) != info.itemsize)
    {
        std::string size_err_msg = "The type you are storing the data in does not contain the same. number of bytes"
                                   "as the type you are storing the data in.";
        throw std::runtime_error(size_err_msg);
    }
    if (info.ndim > dim)
    {
        throw std::runtime_error("Incompatible buffer dimensions");
    }
    std::copy(std::begin(info.shape), std::end(info.shape), std::begin(shape));

    return shape;
}

template <typename num>
auto mat_factory(py_arr<num> &b)
{

    using Scalar = typename std::decay_t<num>;
    auto info = b.request(true);
    auto shape = extract_shape<Scalar, 2>(info);
    const Scalar *const memptr = static_cast<Scalar *>(info.ptr);
    const Mat<Scalar> mat_creator = Mat<Scalar>(memptr, static_cast<arma::uword>(shape[0]),
                                                static_cast<arma::uword>(shape[1]));
    return mat_creator;
}

template <typename num>
auto mat_factory(py_arr<num> &&b)
{

    using Scalar = typename std::decay_t<num>;
    auto info = b.request(true);
    auto shape = extract_shape<Scalar, 2>(info);
    const Scalar *const memptr = static_cast<Scalar *>(info.ptr);
    const Mat<Scalar> mat_creator = Mat<Scalar>(memptr, static_cast<arma::uword>(shape[0]),
                                                static_cast<arma::uword>(shape[1]), true, false);
    return mat_creator;
}

template <typename Scalar>
auto make_col(py::array_t<Scalar> &b)
{

    auto info = b.request(true);
    auto shape = extract_shape<Scalar, 1>(info);
    const Scalar *const memptr = static_cast<Scalar *>(info.ptr);

    const Col<Scalar> return_val(memptr, static_cast<arma::uword>(shape[0]));

    return return_val;
}

template <typename Scalar>
auto make_row(py::array_t<Scalar> &b)
{

    auto info = b.request(true);
    auto shape = extract_shape<Scalar, 1>(info);
    const Scalar *const memptr = static_cast<Scalar *>(info.ptr);

    const arma::Row<Scalar> return_val(memptr, static_cast<arma::uword>(shape[0]));

    return return_val;
}

template <typename num>
auto cube_factory(py_arr_c<num> &b)
{

    auto info = b.request(true);
    auto shape = extract_shape<num, 3>(info);
    auto cube_creator = conv_to<Cube>(static_cast<num *>(info.ptr), shape);
    return cube_creator;
}

template <typename Arr>
auto repr(const Arr &m)
{

    std::stringstream buffer;
    buffer << m;
    return buffer.str();
}

template <typename Scalar>
py::buffer_info def_buffer(arma::Row<Scalar> &&m)
{

    return py::buffer_info(
        m.memptr(), /* data */
        sizeof(Scalar),
        py::format_descriptor<Scalar>::value,
        1,
        py::detail::any_container<py::ssize_t>({m.n_cols}),
        py::detail::any_container<py::ssize_t>({sizeof(Scalar) * static_cast<size_t>(m.n_rows)}));
}

template <typename Scalar>
py::buffer_info def_buffer(Col<Scalar> &&m)
{

    return py::buffer_info(
        m.memptr(),
        sizeof(Scalar),
        py::format_descriptor<Scalar>::value,
        1,
        py::detail::any_container<py::ssize_t>({m.n_rows}),
        py::detail::any_container<py::ssize_t>({sizeof(Scalar)}));
}

template <typename num>
py::buffer_info def_buffer(Mat<num> &&m)
{
    return py::buffer_info(
        m.memptr(),                                                                                         /* Pointer to memory buffer. */
        sizeof(num),                                                                                        /* Size of one scalar. */
        py::format_descriptor<num>::value,                                                                  /*Python struct-style format descriptor */
        2,                                                                                                  /*Number of dimensions */
        py::detail::any_container<py::ssize_t>({m.n_rows, m.n_cols}),                                       /* Shape */
        py::detail::any_container<py::ssize_t>({sizeof(num), (long)(sizeof(num) * m.n_rows)}) /* Strides */ // we have to explicitly define, implicit will pick up the wrong type
    );
}

template <typename num>
py::buffer_info def_buffer(Cube<num> &&m)
{

    return py::buffer_info(
        m.memptr(),
        sizeof(num),
        py::format_descriptor<num>::value,
        3,
        py::detail::any_container<py::ssize_t>({m.n_slices, m.n_rows, m.n_cols}),
        py::detail::any_container<py::ssize_t>({(long)(sizeof(num) * m.n_rows * m.n_cols), sizeof(num), (long)(sizeof(num) * m.n_rows)}));
}

/*                                                                                                                 
 * Converts a possibly                                                                                         
 */
template <typename T>
T clone(T val) { return val; }

template <typename num>
py::buffer_info def_buffer(const Mat<num> &m) { return def_buffer<num>(clone(m)); }

template <typename num>
py::buffer_info def_buffer(const Cube<num> &m) { return def_buffer<num>(clone(m)); }

template <template <typename> typename Arr, typename num>
auto make_py_arr(Arr<num> &&arr)
{
    return py_arr<num>(def_buffer(std::forward<Arr<num>>(arr)));
}

template <template <typename> typename Arr, typename num>
auto make_py_arr(const Arr<num> &arr)
{

    return py_arr<num>(def_buffer(clone(arr)));
}

/*                                                                                                                 
 * This is a templated functor that has overloads that convert the various types that I want to pass from Python   
 * to C++.                                                                                                         
 */
struct make_cpp
{

    template <typename InnerArgType>
    auto operator()(InnerArgType &&x) -> std::enable_if_t<!std::is_pod<InnerArgType>::value,
                                                          decltype(::aw::mat_factory(std::forward<InnerArgType>(x)))>
    {

        return ::aw::mat_factory(std::forward<InnerArgType>(x));
    }

    /* I just pass Plain old data types directly. */
    template <typename InnerArgType>
    auto operator()(InnerArgType x) -> std::enable_if_t<std::is_pod<InnerArgType>::value, InnerArgType>
    {

        return x;
    }
};

/*                                                                                                                 
 * This is a templated functor that has overloads that convert the various types that I want to pass from C++      
 * to Python.                                                                                                      
 */
struct make_py
{

    template <typename InnerArgType>
    auto operator()(InnerArgType &&x) -> std::enable_if_t<!std::is_pod<InnerArgType>::value,
                                                          decltype(::aw::make_py_arr(std::forward<InnerArgType>(x)))>
    {
        return ::aw::make_py_arr(std::forward<InnerArgType>(x));
    }

    template <typename InnerArgType>
    auto operator()(InnerArgType x) -> std::enable_if_t<std::is_pod<InnerArgType>::value, InnerArgType>
    {
        return x;
    }
};

template <typename T1, typename T2>
void assert_same_n_rows(const T1 &first, const T2 &second)
{

    assert(first.n_rows == second.n_rows);
}

template <typename T1, typename T2, typename... Args>
void assert_same_n_rows(const T1 &first, const T2 &second, const Args &... args)
{

    assert_same_n_rows(first, second);
    assert_same_n_rows(second, args...);
}

template <typename T1, typename T2>
void assert_same_n_cols(const T1 &first, const T2 &second)
{

    assert(first.n_cols == second.n_cols);
}

template <typename T1, typename T2, typename... Args>
void assert_same_n_cols(const T1 &first, const T2 &second, const Args &... args)
{

    assert_same_n_cols(first, second);
    assert_same_n_cols(second, args...);
}

template <typename T1, typename T2>
void assert_same_n_slices(const T1 &first, const T2 &second)
{

    assert(first.n_slices == second.n_slices);
}

template <typename T1, typename T2, typename... Args>
void assert_same_n_slices(const T1 &first, const T2 &second, const Args &... args)
{

    assert_same_n_slices(first, second);
    assert_same_n_slices(second, args...);
}

template <typename T1, typename T2>
void assert_same_size(const T1 &first, const T2 &second)
{

    assert(arma::size(first) == arma::size(second));
}

template <typename T1, typename T2, typename... Args>
void assert_same_size(const T1 &first, const T2 &second, const Args &... args)
{

    assert_same_size(first, second);
    assert_same_size(second, args...);
}

template <typename T1, typename T2>
void assert_same(const T1 &first, const T2 &second)
{

    assert(first == second);
}

template <typename T1, typename T2, typename... Args>
void assert_same(const T1 &first, const T2 &second, const Args &... args)
{

    assert(first == second);
    assert_same(second, args...);
}

template <typename ArrayType>
void assert_is_finite(const ArrayType &arr)
{

    assert(arr.is_finite());
}

template <typename ArrayType, typename... Args>
void assert_is_finite(const ArrayType &arr, const Args &... args)
{

    assert_is_finite(arr);
    assert_is_finite(args...);
}

template <typename ArrayType>
void assert_square(const ArrayType &arr)
{

    assert(arr.n_cols == arr.n_rows);
}

template <typename ArrayType, typename... Args>
void assert_square(const ArrayType &arr, const Args &... args)
{

    assert_square(arr);
    assert_square(args...);
}

/*
 * Takes a vector and appends an element to it. 
 */
template <typename NumericType>
Col<NumericType> append(Col<NumericType> arr, const NumericType val)
{

    arr.resize(arr.n_elem + 1);
    arr.tail(1) = val;

    return arr;
}

template <typename NumericType>
arma::Row<NumericType> append(arma::Row<NumericType> arr, const NumericType val)
{

    arr.resize(arr.n_elem + 1);
    arr.tail(1) = val;

    return arr;
}

/* Takes a matrix and appends a row to it. */
template <typename NumericType>
Mat<NumericType> append(Mat<NumericType> arr, const arma::Row<NumericType> row)
{

    assert_same_n_cols(arr, row);
    arr.resize(arr.n_rows + 1, arr.n_cols);
    arr.tail_rows(1) = row;

    return arr;
}

/*Takes a matrix and appends a column to it. */
template <typename NumericType>
arma::Mat<NumericType> append(Mat<NumericType> arr, const Col<NumericType> vec)
{

    assert_same_n_rows(arr, vec);
    arr.resize(arr.n_rows, arr.n_cols + 1);
    arr.tail_cols(1) = vec;

    return arr;
}

/* Appends a matrix to the end of a cube. */
template <typename NumericType>
arma::Cube<NumericType> append(arma::Cube<NumericType> arr, const Mat<NumericType> mat)
{

    assert_same_n_cols(arr, mat);
    assert_same_n_rows(arr, mat);

    arr.resize(arr.n_rows, arr.n_cols, arr.n_slices + 1);
    arr.tail_slice(1) = mat;
}

template <typename NumericType>
void append_inplace(Col<NumericType> &arr, const NumericType val)
{

    arr.resize(arr.n_elem + 1);
    arr.tail(1) = val;
}

template <typename NumericType>
void append_inplace(arma::Row<NumericType> &arr, const NumericType val)
{

    arr.resize(arr.n_elem + 1);
    arr.tail(1) = val;
}

/* Takes a matrix and appends a row to it. */
template <typename NumericType>
void append_inplace(Mat<NumericType> &arr, const arma::Row<NumericType> row)
{

    assert_same_n_cols(arr, row);
    arr.resize(arr.n_rows + 1, arr.n_cols);
    arr.tail_rows(1) = row;
}

/*Takes a matrix and appends a column to it. */
template <typename NumericType>
void append_inplace(Mat<NumericType> &arr, const Col<NumericType> vec)
{

    assert_same_n_rows(arr, vec);
    arr.resize(arr.n_rows, arr.n_cols + 1);
    arr.tail_cols(1) = vec;
}

/* Appends a matrix to the end of a cube. */
template <typename NumericType>
void append_inplace(arma::Cube<NumericType> &arr, const Mat<NumericType> mat)
{

    assert_same_n_cols(arr, mat);
    assert_same_n_rows(arr, mat);

    arr.resize(arr.n_rows, arr.n_cols, arr.n_slices + 1);
    arr.slice(arr.n_slices - 1) = mat;
}

/* 
 * Converts a container to a scalar. It calls the as_scalar funciton in Armadillo if it exists. 
 */
template <typename ContainerType>
auto as_scalar(ContainerType &&container) -> decltype(arma::as_scalar(std::forward<ContainerType>(container)))
{

    return arma::as_scalar(std::forward<ContainerType>(container));
}

/* 
 * I provide a template specialization for std::vector. I convert to an Armadillo vector and then call 
 * the std::vector there. By doing this I can recover the same exceptions as in the case above. 
 */
template <typename ValueType>
ValueType as_scalar(const std::vector<ValueType> &container)
{

    return arma::as_scalar(Col<ValueType>(container));
}

/* 
 * Converts an intmat to an unsigned long vector. 
 */
iuvec unsigned_vectorize(const intmat &matrix)
{

    iuvec unsigned_vec(matrix.size());
    std::copy(cbegin(matrix), cend(matrix), begin(unsigned_vec));

    return unsigned_vec;
}

/* 
 * Converts a vector to its square matrix as long as the vector has a number of elmeents that is a perfect square.
 */
template <typename ValueType>
Mat<ValueType> reshape_square(Col<ValueType> &&vec)
{

    npulong side_length = (npulong)std::sqrt(vec.n_elem);
    assert(side_length * side_length == vec.n_elem && "The vector does not have a perfect square number of "
                                                      "elements");
    return arma::reshape(std::forward<Col<ValueType>>(vec), side_length, side_length);
}

template <typename ValueType>
Mat<ValueType> reshape_square(const std::vector<ValueType> &vec)
{
    return reshape_square(Col<ValueType>(vec));
}

} // namespace aw
#endif /* ARMA_WRAPPER_H */
