/*
 * @FilePath: \SHRED_GEDI\code\methods\pointnet2\rebuild\pointnet2\_ext-src\src\bindings.cpp
 * @Author: AceSix
 * @Date: 2022-10-27 12:35:11
 * @LastEditors: AceSix
 * @LastEditTime: 2022-10-27 16:13:19
 * Copyright (C) 2022 Brown U. All rights reserved.
 */
#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK 
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points", &gather_points);
  m.def("gather_points_grad", &gather_points_grad);
  m.def("furthest_point_sampling", &furthest_point_sampling);

  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);

  m.def("ball_query", &ball_query);

  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
}
