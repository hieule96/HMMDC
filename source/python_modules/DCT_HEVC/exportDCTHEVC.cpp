#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "TLibCommon/TComRom.h"
#include "Transform.h"
namespace py=pybind11;

py::array_t<int> dct2D (py::array_t <int> block){
    py::buffer_info b_input = block.request();
    if (b_input.format!=py::format_descriptor<int>::format())
    {
        throw std::runtime_error("Incompatible format: expected a int array!");
    }
    int w = b_input.shape[1];
    int h = b_input.shape[0];
    py::array_t<int> result = py::array_t<int>(b_input.size);
    py::buffer_info b_result = result.request();
    xTr(8,(int *)b_input.ptr,(int*)b_result.ptr,w,w,true,15);
    result.resize({w,h});
    //reshape again to 2D
    return result;
}
py::array_t<int> idct2D (py::array_t <int> block){
    py::buffer_info b_input = block.request();
    int w = b_input.shape[1];
    int h = b_input.shape[0];
    py::array_t<int> result = py::array_t<int>(b_input.size);
    py::buffer_info b_result = result.request();
    xITr(8,(int *)b_input.ptr,(int *)b_result.ptr,w,w,true,15);
    //reshape again to 2D
    result.resize({w,h});
    return result;
}
PYBIND11_MODULE(Transform_HEVC,m){
    py::class_<Quant>(m,"Quant")
        .def(py::init<>())
        .def("quantize2D",&Quant::quantize2D)
        .def("getAbsSum",&Quant::getAbsSum)
        .def("dequantize2D",&Quant::dequantize2D);
    m.def("dct2D",&dct2D);
    m.def("idct2D",&idct2D);
    m.def("initROM",&initROM);
}