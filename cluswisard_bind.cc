#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cluswisard.cc"

namespace py = pybind11;

PYBIND11_MODULE(wisard, m)
{
    py::class_<ClusWisard>(m, "ClusWisard")
      .def(py::init<int, float, int>())
      .def("train", (void (ClusWisard::*)(const vector<vector<int>>&, const vector<string>&)) &ClusWisard::train)
      .def("classify", (vector<string>& (ClusWisard::*)(const vector<vector<int>>&)) &ClusWisard::classify)
    ;
}
