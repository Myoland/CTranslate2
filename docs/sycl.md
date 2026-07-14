# Intel SYCL backend

The SYCL backend runs CTranslate2 models on Intel GPUs through the Level Zero
runtime. It is currently a source-build feature for Linux x86-64 and is not
included in the prebuilt Python wheels.

## Supported configuration

The initial backend requires Intel oneAPI DPC++/C++ Compiler and oneMKL 2026.1
or newer. It supports standard CTranslate2 models with `float32`, `float16`,
`bfloat16`, `int8`, `int8_float32`, `int8_float16`, and `int8_bfloat16`
computation. AWQ, FlashAttention 2, and tensor parallelism are not supported.

SYCL cannot be combined with CUDA or HIP in the same library. A SYCL build
still includes the CPU backend. Only shared-library builds are currently
supported (`BUILD_SHARED_LIBS=ON`).

## Install the toolchain

Install the Intel Level Zero compute runtime for the distribution. On Ubuntu
26.04, follow Intel's [client GPU installation
guide](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html)
and install `libze-intel-gpu1`, `libze1`, and `intel-ocloc`.

Configure Intel's [oneAPI APT
repository](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-apt.html),
then install the compiler and oneMKL development packages:

```bash
sudo apt-get install intel-oneapi-compiler-dpcpp-cpp-2026.1 \
                     intel-oneapi-mkl-devel-2026.1
source /opt/intel/oneapi/setvars.sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls
```

`sycl-ls` should list an Intel GPU before configuring CTranslate2.

## Build

The portable build embeds generic SPIR-V and lets the driver compile kernels
for the visible GPU:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install-sycl" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DWITH_SYCL=ON \
  -DSYCL_TARGETS=spir64 \
  -DWITH_CUDA=OFF -DWITH_HIP=OFF \
  -DWITH_MKL=ON -DOPENMP_RUNTIME=INTEL
cmake --build build-sycl --parallel
cmake --install build-sycl
```

For an Arc B580-specific ahead-of-time build, replace the target option with:

```text
-DSYCL_TARGETS=spir64_gen -DSYCL_AOT_DEVICE=bmg-g21
```

## Python wheel and runtime

Build a local wheel against the installed C++ library:

```bash
python -m venv .venv-sycl
source .venv-sycl/bin/activate
python -m pip install build
CTRANSLATE2_ROOT="$PWD/install-sycl" CXX=icpx python -m build --wheel python
python -m pip install python/dist/*.whl
```

The wheel intentionally uses the external oneAPI runtime. Source oneAPI and
make the CTranslate2 install library visible before importing it:

```bash
source /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH="$PWD/install-sycl/lib:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

Use `sycl` as the canonical device name. `xpu` is accepted as an input alias:

```python
import ctranslate2

print(ctranslate2.get_sycl_device_count())
print(ctranslate2.get_supported_compute_types("sycl"))

model = ctranslate2.models.Whisper(
    "whisper-tiny-ct2",
    device="sycl",
    compute_type="float16",
)
```

SYCL `StorageView` objects can be copied back to CPU, but do not expose the
NumPy or CUDA array interfaces. Zero-copy Python USM and DLPack exchange are
not implemented. The caching allocator retains up to 512 MiB by default; use
`CT2_SYCL_CACHING_ALLOCATOR_MAX_BYTES` to set a different byte limit.
