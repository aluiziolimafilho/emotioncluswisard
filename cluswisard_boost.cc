#include <boost/python.hpp>
#include "cluswisard.cc"

BOOST_PYTHON_MODULE(cluswisard)
{
    using namespace boost::python;
    class_<ClusWisard>("ClusWisard", init<int, float, int>())
      .def("train", &ClusWisard::train)
      .def("classify", &ClusWisard::classify)
}
