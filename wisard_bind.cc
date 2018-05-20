#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "wisard.cc"

namespace py = pybind11;

PYBIND11_MODULE(wisard, m)
{
    py::class_<ClusWisard>(m, "ClusWisard")
      .def(py::init<int, float, int>())
      .def("train", (void (ClusWisard::*)(const vector<vector<int>>&, const vector<string>&)) &ClusWisard::train)
      .def("classify", (vector<string>& (ClusWisard::*)(const vector<vector<int>>&)) &ClusWisard::classify)
      .def_property("verbose", &ClusWisard::getVerbose, &ClusWisard::setVerbose)
    ;

    py::class_<Wisard>(m, "Wisard")
      .def(py::init<int>())
      .def("train", (void (Wisard::*)(const vector<vector<int>>&, const vector<string>&)) &Wisard::train)
      .def("classify", (vector<string>& (Wisard::*)(const vector<vector<int>>&)) &Wisard::classify)
      .def_property("verbose", &Wisard::getVerbose, &Wisard::setVerbose)
    ;
}
