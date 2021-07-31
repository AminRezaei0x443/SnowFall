from setuptools import setup, find_packages

setup(
    name='snowfall-ml',
    version='0.1.0',
    description='Machine Learning Framework based on PyTorch with focus on Neural Networks',
    license='MIT',
    packages=find_packages(),
    author='Amin Rezaei',
    author_email='AminRezaei0x443@gmail.com',
    keywords=['snowfall-ml', 'snowfall', 'neural-networks', 'pytorch'],
    url='https://github.com/aminrezaei0x443/snowfall', install_requires=['torch', 'numpy', 'tqdm']
)
