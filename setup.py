from setuptools import setup, find_packages

setup(
    name='snowfall',
    version='0.0.1',
    description='Machine Learning Platform based on PyTorch with focus on Neural Networks',
    license='Exc',
    packages=find_packages(),
    author='Amin Rezaei',
    author_email='Amin.Rezaei1379@gmail.com',
    keywords=['snowfall', 'neural-networks', 'pytorch'],
    url='https://github.com/aminrezaei0x443/snowfall', install_requires=['torch', 'numpy', 'tqdm']
)
