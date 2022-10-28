/*
 * @FilePath: \SHRED_GEDI\code\methods\pointnet2\rebuild\pointnet2\_ext-src\src\ball_query.cpp
 * @Author: AceSix
 * @Date: 2022-10-27 12:35:11
 * @LastEditors: AceSix
 * @LastEditTime: 2022-10-27 16:07:06
 * Copyright (C) 2022 Brown U. All rights reserved.
 */
#include "ball_query.h"
#include "utils.h"

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK 
#endif

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return idx;
}
