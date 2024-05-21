import glob
import os
import time
from itertools import chain

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SRC_DIR = os.path.abspath(os.path.dirname(__file__))


def get_cpp_or_cuda_sources(src_dir):
    files = glob.glob(f'{src_dir}/*.cu') + glob.glob(f'{src_dir}/*.cpp')
    print(f'\033[31mFind {len(files)} cu/cpp files in directory: {src_dir}\033[0m')
    return files


setup(
    name='extension',
    version='2022.11',
    description='build time {}'.format(time.strftime("%y-%m-%d %H:%M:%S", time.localtime(time.time()))),
    ext_modules=[
        CUDAExtension(
            name='_C',
            sources=list(
                chain(
                    get_cpp_or_cuda_sources('src'),

                    get_cpp_or_cuda_sources('src/other'),
                    get_cpp_or_cuda_sources('src/ops_3d'),
                    get_cpp_or_cuda_sources('src/nerf'),
                )
            ),
            extra_compile_args={
                'cxx': ["-fopenmp", "-O3"],
                'nvcc': [
                    '-O3',
                    # '-rdc=true',
                    # '--ptxas-options=-v',
                ]
            },
            define_macros=[("CUDA_HAS_FP16", "1"), ("__CUDA_NO_HALF_OPERATORS__", None)],
            include_dirs=[
                os.path.join(SRC_DIR, "include"),
                os.path.join(SRC_DIR, "third_party/glm"),
            ],
            # libraries=[],
            # library_dirs=[]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
