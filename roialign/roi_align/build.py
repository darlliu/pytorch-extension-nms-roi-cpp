from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup (
    name = 'roicuda',
    ext_modules = [
        CUDAExtension(
            name = "roicuda",
            sources = ["./roicpp.cpp","./roicppkernel.cu"],
            extra_compile_args = {'cxx':["-DMS_WIN64","-MD" ], "nvcc":["-O2"]},
            include_dirs = [
            "D:/Python37/Lib/site-packages/torch/lib/include",
            "D:/VisualStudio/VS/VC/Tools/MSVC/14.15.26726/include",
            "D:/Windows Kits/10/Include/10.0.17134.0/ucrt",
            "D:/Windows Kits/10/Include/10.0.17134.0/shared"],
            library_dirs = [
            "D:/Python37/Lib/site-packages/torch/lib",
            "D:/Windows Kits/10/Lib/10.0.17134.0/ucrt/x64",
            "D:/Windows Kits/10/Lib/10.0.17134.0/um/x64",
            "D:/VisualStudio/VS/VC/Tools/MSVC/14.15.26726/lib/x64"],
        )
    ],
    cmdclass = {
        "build_ext":BuildExtension
    }
)