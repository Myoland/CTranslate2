#pragma once

#include "activation.h"
#include "add.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class BiasAdd : public Op {
    public:
      BiasAdd(const ActivationType* activation_type = nullptr, const dim_t axis = -1);

      void operator()(const StorageView& value,
                      const StorageView& bias,
                      StorageView& output,
                      const StorageView* residual = nullptr) const;

      // Adds a last-dimension bias and scatters 3 equal parts in one kernel.
      // This specialized path is used by iterative FP16 QKV projections.
      void split3(const StorageView& value,
                  const StorageView& bias,
                  StorageView& output1,
                  StorageView& output2,
                  StorageView& output3) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& value,
                   const StorageView& bias,
                   StorageView& output,
                   const StorageView* residual) const;

      template <typename T>
      void compute_split3(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output1,
                          StorageView& output2,
                          StorageView& output3) const;

      const ActivationType* _activation_type;
      const dim_t _axis;
    };

  }
}
