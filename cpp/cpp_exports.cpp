#include <iostream>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <stdint.h>
#include <utility>
#include <numpy/npy_common.h>
#include <functional>
#include "arma_wrapper/arma_wrapper.h"
#include "arma_wrapper/function_wrapper.h"
#include "arma_wrapper/armadillo_sparse.h"
#include <pybind11/stl.h>
#include "utilities.h"

namespace py = pybind11;

using arma::Cube;
using arma::Mat;
using aw::py_arr;
using aw::py_arr_c;

using aw::dcube;
using aw::dmat;
using aw::intcube;
using aw::intmat;
using aw::npdouble;
using aw::npint;
using dvec = arma::Col<npdouble>;

// Declearation
py::dict gen_solver(arma::mat B,
                    py::function f,
                    py::function g,
                    int useg,
                    double rho,
                    double eta,
                    double gamma,
                    double tau,
                    double epsilon,
                    double btol,
                    double ftol,
                    double gtol,
                    int maxitr,
                    int verbose);

double local_f(const arma::mat &B,
               const arma::mat &X,
               const arma::mat &Y,
               double bw,
               int ncore);

py::dict local_solver(arma::mat B,
                      arma::mat &X,
                      arma::mat &Y,
                      double bw,
                      double rho,
                      double eta,
                      double gamma,
                      double tau,
                      double epsilon,
                      double btol,
                      double ftol,
                      double gtol,
                      int maxitr,
                      int verbose,
                      int ncore);

double phd_init(const arma::mat &B,
                const arma::mat &X,
                const arma::mat &Y,
                double bw,
                int ncore);

py::dict phd_solver(arma::mat B,
                    arma::mat &X,
                    arma::mat &Y,
                    double bw,
                    double rho,
                    double eta,
                    double gamma,
                    double tau,
                    double epsilon,
                    double btol,
                    double ftol,
                    double gtol,
                    int maxitr,
                    int verbose,
                    int ncore);

double save_init(const arma::mat &B,
                 const arma::mat &X,
                 const arma::mat &Y,
                 double bw,
                 int ncore);

py::dict save_solver(arma::mat B,
                     arma::mat &X,
                     arma::mat &Y,
                     double bw,
                     double rho,
                     double eta,
                     double gamma,
                     double tau,
                     double epsilon,
                     double btol,
                     double ftol,
                     double gtol,
                     int maxitr,
                     int verbose,
                     int ncore);

double seff_init(const arma::mat &B,
                 const arma::mat &X,
                 const arma::mat &Y,
                 double bw,
                 int ncore);

py::dict seff_solver(arma::mat B,
                     arma::mat &X,
                     arma::mat &Y,
                     double bw,
                     double rho,
                     double eta,
                     double gamma,
                     double tau,
                     double epsilon,
                     double btol,
                     double ftol,
                     double gtol,
                     int maxitr,
                     int verbose,
                     int ncore);

double sir_init(const arma::mat &B,
                const arma::mat &X,
                const arma::mat &Y,
                double bw,
                int ncore);

py::dict sir_solver(arma::mat B,
                    arma::mat &X,
                    arma::mat &Y,
                    double bw,
                    double rho,
                    double eta,
                    double gamma,
                    double tau,
                    double epsilon,
                    double btol,
                    double ftol,
                    double gtol,
                    int maxitr,
                    int verbose,
                    int ncore);

py::dict surv_dm_solver(arma::mat B,
                        const arma::mat &X,
                        const arma::mat &Phit,
                        const arma::vec &Fail_Ind,
                        double bw,
                        double rho,
                        double eta,
                        double gamma,
                        double tau,
                        double epsilon,
                        double btol,
                        double ftol,
                        double gtol,
                        int maxitr,
                        int verbose,
                        int ncore);

py::dict surv_dn_solver(arma::mat B,
                        const arma::mat &X,
                        const arma::mat &Phit,
                        const arma::vec &Fail_Ind,
                        double bw,
                        double rho,
                        double eta,
                        double gamma,
                        double tau,
                        double epsilon,
                        double btol,
                        double ftol,
                        double gtol,
                        int maxitr,
                        int verbose,
                        int ncore);

py::dict surv_forward_solver(arma::mat B,
                             const arma::mat &X,
                             const arma::vec &Fail_Ind,
                             double bw,
                             double rho,
                             double eta,
                             double gamma,
                             double tau,
                             double epsilon,
                             double btol,
                             double ftol,
                             double gtol,
                             int maxitr,
                             int verbose,
                             int ncore);

int main()
{
    std::cout << "compile success, can use function" << std::endl;
    return 0;
}

PYBIND11_MODULE(cpp_exports, m)
{
    m.doc() = "orthoDr"; // optional module docstring

    // numpy -> armadillo
    py::class_<dmat>(m, "mat", py::buffer_protocol())
        .def(py::init([](py_arr<npdouble> &arr) { return dmat(aw::mat_factory(arr)); }), "Creates a mat from an ndarray.")
        .def("__repr__", [](const dmat &x) { return aw::repr(x); }, "Uses Armadillo's print formatting.")
        .def_buffer([](dmat &x) { return aw::def_buffer<npdouble>(std::move(x)); }); // here we pass in a arma::Mat<double>, std::move is to keep content, so its like a shallow copy, this is move sementic

    py::class_<intmat>(m, "intmat", py::buffer_protocol())
        .def(py::init([](py_arr<npint> &arr) { return intmat(aw::mat_factory(arr)); }), "Creates a intmat from an ndarray.")
        .def("__repr__", [](const intmat &x) { return aw::repr(x); }, "Uses Armadillo's print formatting.")
        .def_buffer([](intmat &x) { return aw::def_buffer<npint>(std::move(x)); });

    py::class_<dcube>(m, "cube", py::buffer_protocol())
        .def(py::init([](py_arr_c<npdouble> &arr) { return dcube(aw::cube_factory(arr)); }), "Create a Cube<np_float> from an ndarray.")
        .def("__repr__", [](const dcube &x) { return aw::repr(x); }, "Uses Armadillo's print formatting.")
        .def_buffer([](dcube &x) { return aw::def_buffer<npdouble>(std::move(x)); });

    py::class_<intcube>(m, "intcube", py::buffer_protocol())
        .def(py::init([](py_arr_c<npint> &arr) { return intcube(aw::cube_factory(arr)); }), "Create a intcube from an ndarray.")
        .def("__repr__", [](const intcube &x) { return aw::repr(x); }, "Uses Armadillo's print formatting.")
        .def_buffer([](intcube &x) { return aw::def_buffer<npint>(std::move(x)); });

    py::implicitly_convertible<py_arr<npdouble>, dmat>();
    py::implicitly_convertible<py_arr<npint>, intmat>();
    py::implicitly_convertible<py_arr<npint>, intcube>();
    py::implicitly_convertible<py_arr<npdouble>, dcube>();

    // function export
    m.def("_gen_solver", &gen_solver, "orthodr export function gen_solver");
    m.def("_local_f", &local_f, "orthodr export function local_f");
    m.def("_local_solver", &local_solver, "orthodr export function local_solver");
    m.def("_phd_init", &phd_init, "orthodr export function phd_init");
    m.def("_phd_solver", &phd_solver, "orthodr export function phd_solver");
    m.def("_save_init", &save_init, "orthodr export function save_init");
    m.def("_save_solver", &save_solver, "orthodr export function save_solver");
    m.def("_seff_init", &seff_init, "orthodr export function seff_init");
    m.def("_seff_solver", &seff_solver, "orthodr export function seff_solver");
    m.def("_sir_init", &sir_init, "orthodr export function sir_init");
    m.def("_sir_solver", &sir_solver, "orthodr export function sir_solver");
    m.def("_surv_dm_solver", &surv_dm_solver, "orthodr export function surv_dm_solver");
    m.def("_surv_dn_solver", &surv_dn_solver, "orthodr export function surv_dn_solver");
    m.def("_surv_forward_solver", &surv_forward_solver, "orthodr export function surv_forward_solver");
    m.def("_KernelDist_cross", &KernelDist_cross, "orthodr export function KernelDist_cross");

    // test functions
    m.def("main", &main, "Sums2 the elements in the array.");
}
