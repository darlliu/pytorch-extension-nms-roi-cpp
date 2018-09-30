import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from roicuda import gpu_roi_forward,gpu_roi_backward


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = None 

        if image.is_cuda:
            crops = gpu_roi_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width)
        else:
            raise NotImplementedError("CPU version is currently not supported")
    
        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)
        # print ("got crops back into python, shape {}".format(crops.shape))
        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        if grad_outputs.is_cuda:
            # print ("about to backprop grads_output with shape {}, grads_mage with shape {}".format(grad_outputs.shape, grad_image.shape))
            grad_image = gpu_roi_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        else:
            raise NotImplementedError("CPU version is currently not supported")

        return grad_image, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)
