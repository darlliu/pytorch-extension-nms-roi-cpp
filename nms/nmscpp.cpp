// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// Ported to C++, by YL
// ------------------------------------------------------------------
#include <math.h>
#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Type.h>

typedef float scalar_t ;
typedef int64_t intlike_t ;

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(intlike_t) * 8;
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void _nms(int boxes_num, scalar_t * boxes_dev,
          intlike_t * mask_dev, float nms_overlap_thresh);
int gpu_nms(
    at::Tensor& keep,
    at::Tensor& num_out, 
    at::Tensor& boxes, 
    float nms_overlap_thresh
    )
{
CHECK_INPUT (boxes);

// Number of ROIs
auto boxes_num = boxes.size(0);
auto boxes_dim = boxes.size(1);

auto boxes_flat = boxes.data<scalar_t>();

const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

auto mask = at::CUDA(at::kLong).zeros({boxes_num, col_blocks});
auto mask_flat = mask.data<intlike_t>();
_nms(boxes_num, boxes_flat, mask_flat, nms_overlap_thresh);


std :: cout << "completed the cuda kernel "<<std::endl;
auto mask_cpu = mask.toBackend(at::Backend::CPU);
auto mask_cpu_flat = mask_cpu.data<intlike_t>();

auto remv_cpu = at::CPU(at::kLong).zeros({col_blocks});
auto remv_cpu_flat = remv_cpu.data<intlike_t>();

auto keep_flat = keep.data<intlike_t>();
intlike_t num_to_keep = 0;

std :: cout << "completed setting up keep_flat "<<std::endl;
int i, j;
for (i = 0; i < boxes_num; i++) {
int nblock = i / threadsPerBlock;
int inblock = i % threadsPerBlock;

if (!(remv_cpu_flat[nblock] & (1ULL << inblock))) {
    keep_flat[num_to_keep++] = i;
    auto p = &mask_cpu_flat[0] + i * col_blocks;
    for (j = nblock; j < col_blocks; j++) {
    remv_cpu_flat[j] |= p[j];
    }
}
}
std :: cout << "completed most work, got this back for number to keep "<<num_to_keep <<std::endl;
auto num_out_flat = num_out.data<intlike_t>();
* num_out_flat = num_to_keep;

return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gpu_nms", &gpu_nms, "NMS with CUDA kernel");
}