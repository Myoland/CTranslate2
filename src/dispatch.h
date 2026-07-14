#pragma once

#include "device_dispatch.h"
#include "type_dispatch.h"

#define DEVICE_AND_TYPE_DISPATCH(DEVICE, TYPE, STMTS)   \
  DEVICE_DISPATCH(DEVICE, TYPE_DISPATCH(TYPE, (STMTS)))


#define NON_FLOAT_CASE(NAME)                                            \
  default:                                                              \
    throw std::invalid_argument(NAME " only supports float types");     \


#if defined(CT2_WITH_CUDA) && defined(CT2_WITH_SYCL)
#  define REDUCED_FLOAT_DEVICE_DISPATCH(NAME, DEVICE, STMTS)       \
  switch (DEVICE) {                                                \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))                   \
    DEVICE_CASE(Device::SYCL, SINGLE_ARG(STMTS))                   \
    case Device::CPU:                                              \
      throw std::invalid_argument(NAME " with reduced precision is only supported on accelerators"); \
  }
#elif defined(CT2_WITH_CUDA)
#  define REDUCED_FLOAT_DEVICE_DISPATCH(NAME, DEVICE, STMTS)       \
  switch (DEVICE) {                                                \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))                   \
    UNSUPPORTED_DEVICE_CASE(Device::SYCL)                          \
    case Device::CPU:                                              \
      throw std::invalid_argument(NAME " with reduced precision is only supported on accelerators"); \
  }
#elif defined(CT2_WITH_SYCL)
#  define REDUCED_FLOAT_DEVICE_DISPATCH(NAME, DEVICE, STMTS)       \
  switch (DEVICE) {                                                \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)                          \
    DEVICE_CASE(Device::SYCL, SINGLE_ARG(STMTS))                   \
    case Device::CPU:                                              \
      throw std::invalid_argument(NAME " with reduced precision is only supported on accelerators"); \
  }
#else
#  define REDUCED_FLOAT_DEVICE_DISPATCH(NAME, DEVICE, STMTS)       \
  switch (DEVICE) {                                                \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)                          \
    UNSUPPORTED_DEVICE_CASE(Device::SYCL)                          \
    case Device::CPU:                                              \
      throw std::invalid_argument(NAME " with reduced precision is only supported on accelerators"); \
  }
#endif

#define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)            \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    TYPE_CASE(float16_t, REDUCED_FLOAT_DEVICE_DISPATCH(NAME, DEVICE, (STMTS))) \
    TYPE_CASE(bfloat16_t, REDUCED_FLOAT_DEVICE_DISPATCH(NAME, DEVICE, (STMTS))) \
    NON_FLOAT_CASE(NAME)                                                \
  }
