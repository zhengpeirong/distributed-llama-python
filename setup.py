from setuptools import setup, Extension

quants_ext = Extension(
    "distributed_llama.quants._quants",
    sources=["distributed_llama/quants/quants.c"],
    extra_compile_args=["-O3", "-fPIC", "-march=native"],
)

ops_ext = Extension(
    "distributed_llama.ops._ops",
    sources=[
        "distributed_llama/ops/ops.c",
        "distributed_llama/ops/sgemm_wrapper.cpp",
    ],
    extra_compile_args=["-O3", "-fPIC", "-march=native"],
    language="c++",
)

setup(
    ext_modules=[quants_ext, ops_ext],
)
