# pytorch-extension-nms-roi-cpp
C++ port of nms and roi_align extensions (CUDA ver.) compilable in Windows 10 with VS2017

An excercise to port 2 pytorch C extensions to C++ and compatible with Win10+VS2017. May contain bugs. For use with 

https://github.com/multimodallearning/pytorch-mask-rcnn

go into each folder and run ```python build.py install``` changing include_dirs and include_libs to local VS2017 directories.

now copy paste the whole folder including python files to pytorch-mask-rcnn and it should work (only cuda version is ported so cuda is needed).

Result from running demo.py using the two extensions :

![Test Demo.py](./demofig.png)
