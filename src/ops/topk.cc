#include "ctranslate2/ops/topk.h"

#include <limits>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    TopK::TopK(dim_t k, dim_t axis)
      : _k(k) {
      if (axis != -1)
        throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
    }

    void TopK::operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
      PROFILE("TopK");
      if (x.rank() == 0)
        throw std::invalid_argument("TopK requires an input with at least 1 dimension");

      const dim_t depth = x.dim(-1);
      if (_k < 0 || _k > depth)
        throw std::invalid_argument("TopK k must be between 0 and the last input dimension");
      if (depth > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("TopK input depth exceeds the INT32 index range");

      dim_t batch_size = 1;
      for (dim_t axis = 0; axis + 1 < x.rank(); ++axis)
        batch_size *= x.dim(axis);
      values.resize({batch_size, _k});
      indices.resize({batch_size, _k});

      if (_k == 0 || batch_size == 0)
        return;

      DEVICE_AND_FLOAT_DISPATCH("TopK", x.device(), x.dtype(),
                                (compute<D, T, int32_t>(x, values, indices)));
    }

  }
}
