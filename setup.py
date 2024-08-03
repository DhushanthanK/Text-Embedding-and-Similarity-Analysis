from setuptools import setup, find_packages

setup(
    name='embedding_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'datasets',
        'transformers',
        'torch',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'matplotlib',
        'gensim',
        'sentence_transformers',
    ],
)
