import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension



sources = glob.glob('src/*.cpp')+glob.glob('src/*.cu')


setup(
    name='my_lib',
    version='0.1',
    author='adamdad',
    author_email='yxy_adadm@qq.com',
    description='A test project',
    long_description='',
    ext_modules=[
        CppExtension('my_lib', sources)
    ],
    cmdclass={'build_ext': BuildExtension}
)