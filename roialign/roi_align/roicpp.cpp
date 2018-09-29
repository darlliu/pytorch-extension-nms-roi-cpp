#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Type.h>
#include <iostream>


void CropAndResizeBackpropImageLaucher(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr); 



void CropAndResizeLaucher(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr);


at::Tensor crop_and_resize_gpu_forward(
    at::Tensor image,
    at::Tensor boxes,           // [y1, x1, y2, x2]
    at::Tensor box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width
) {
    const int batch_size = image.size(0);
    const int depth = image.size(1);
    const int image_height = image.size(2);
    const int image_width = image.size(3);

    const int num_boxes = boxes.size(0);
    // init output space
    auto crops = torch::CUDA(at::kFloat).zeros({num_boxes, depth, crop_height, crop_width});
    CropAndResizeLaucher(
        image.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value,
        crops.data<float>()
    );
    // auto crops = crops_cuda.toBackend(at::Backend::CPU);
    // std::cout << "roi ended, got crops shape "<<crops.sizes()<<std::endl;
    return crops;
}


at::Tensor crop_and_resize_gpu_backward(
    at::Tensor grads,
    at::Tensor boxes,      // [y1, x1, y2, x2]
    at::Tensor box_index,    // range in [0, batch_size)
    at::Tensor grads_image // resize to [bsize, c, hc, wc], CPU
) {
    // shape
    std::cout << "about to launch backprop roi"<<std::endl;
    const int batch_size = grads_image.size(0);
    const int depth = grads_image.size(1);
    const int image_height = grads_image.size(2);
    const int image_width = grads_image.size(3);

    const int num_boxes = grads.size(0);
    const int crop_height = grads.size(2);
    const int crop_width = grads.size(3);

    // init output space
    grads_image *= 0 ;
    auto grads_cuda = torch::CUDA(at::kFloat).zeros_like(grads_image);
    CropAndResizeBackpropImageLaucher(
        grads.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth,
        grads_cuda.data<float>()
    );
    return grads_cuda; //.toBackend(at::Backend::CPU);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gpu_roi_forward", &crop_and_resize_gpu_forward, "NMS with CUDA kernel");
  m.def("gpu_roi_backward", &crop_and_resize_gpu_backward, "NMS with CUDA kernel");
}