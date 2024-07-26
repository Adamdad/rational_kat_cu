import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension



sources = glob.glob('src/*.cpp')+glob.glob('src/*.cu')


setup(
    name='my_lib',
    version='0.1',
    author='adamdad',
    author_email='yxy_adadm@qq.com',
    description='A test project',
    long_description='',
    ext_modules=[
        CUDAExtension(name='my_lib', 
                      sources=sources
                      )
    ],
    cmdclass={'build_ext': BuildExtension}
)