#ifndef ARMA_WRAPPER_ARMADILLO_H
#define ARMA_WRAPPER_ARMADILLO_H

#include <type_traits>
#include <algorithm>
#include <iterator>
#include <pybind11/numpy.h>
#include <armadillo>
#include "function_wrapper.h"

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename T>
using is_SpMat = aw::is_specialization_of<arma::SpMat, T>;
template <typename T>
using is_Col = aw::is_specialization_of<arma::Col, T>;
template <typename T>
using is_Row = aw::is_specialization_of<arma::Row, T>;
using std::enable_if_t;

/* 
 * HAndles conversions from array to Column vectors.  
 */
// template <typename Type>
// struct type_caster<Type, enable_if_t<is_Col<Type>::value || is_Row<Type>::value>>
// {

//     using Scalar = typename std::decay_t<typename Type::elem_type>;

//     /*
//      * This function loads python arrays and coverts them to Armadillo columns.
//      */

//     bool load(handle src, bool)
//     {

//         array_t<Scalar> buf(src, true);
//         if (!buf.check())
//         {
//             return false;
//         }

//         if (is_Col<Type>::value)
//         {
//             value = aw::make_col(buf);
//         }
//         else
//         {

//             value = aw::make_row(buf);
//         }

//         return true;
//     }

//     static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */)
//     {

//         auto returnval = aw::make_py_arr(src).release();

//         return returnval;
//     }

//     PYBIND11_TYPE_CASTER(Type, _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name());
//     //                                   _("[]") + n_rows + _(", ") + n_cols + _("]]"));
// };

template <typename Type>
struct type_caster<Type, enable_if_t<is_SpMat<Type>::value>>
{

    using Scalar = typename std::decay_t<typename Type::elem_type>;
    using StorageIndexType = typename std::remove_reference<decltype(*std::declval<Type>().row_indices)>::type;
    using IndexType = typename std::remove_reference<decltype(std::declval<Type>().n_nonzero)>::type;

    /* 
     * This function transfers python sparse arrays to armadillo. Armadillo only accepts csc format matrices, 
     * and do it converts them if necessary. 
     */
    bool load(handle src, bool)
    {
        if (!src)
        {

            return false;
        }

        object obj(src, true);
        object sparse_module = module::import("scipy.sparse");
        object matrix_type = sparse_module.attr("csc_matrix");

        if (obj.get_type() != matrix_type.ptr())
        {

            try
            {
                obj = matrix_type(obj);
            }
            catch (const error_already_set &)
            {
                return false;
            }
        }

        auto values = aw::py_arr<Scalar>((object)obj.attr("data"));
        auto row_indices = aw::py_arr<StorageIndexType>((object)obj.attr("indices"));
        auto col_ptrs = aw::py_arr<StorageIndexType>((object)obj.attr("indptr"));
        auto shape = pybind11::tuple((pybind11::object)obj.attr("shape"));

        if (!values.check() || !row_indices.check() || !col_ptrs.check())
        {
            return false;
        }

        value = arma::SpMat<Scalar>(arma::conv_to<arma::uvec>::from(aw::make_cpp()(row_indices)),
                                    arma::conv_to<arma::uvec>::from(aw::make_cpp()(col_ptrs)),
                                    arma::vectorise(aw::make_cpp()(values)),
                                    shape[0].cast<IndexType>(),
                                    shape[1].cast<IndexType>());

        return true;
    }

    /* 
     * This function transfers the data from the csc format in armadillo to the equivalenet format in 
     * python. 
     */
    static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */)
    {

        object matrix_type = module::import("scipy.sparse").attr("csc_matrix");

        // Armadillo pads the arrays with an extra elements, and so we only copy the
        // elements that are necessary in Python.
        array data(static_cast<size_t>(src.n_nonzero), src.values);
        array col_ptrs(static_cast<size_t>(src.n_cols + 1), src.col_ptrs);
        array row_indices(static_cast<size_t>(src.n_nonzero), src.row_indices);

        return matrix_type(
                   std::make_tuple(data, row_indices, col_ptrs),
                   std::make_pair(static_cast<aw::npulong>(src.n_rows), static_cast<aw::npulong>(src.n_cols)))
            .release();
    }

    PYBIND11_TYPE_CASTER(Type, _("scipy.sparse.csc_matrix[") + npy_format_descriptor<std::decay_t<Scalar>>::name() + _("]"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

#endif /* ARMA_WRAPPER_ARMADILL_H */
