from setuptools import find_packages, setup

setup(
    name='text2props',
    packages=find_packages(),
    version='0.2.0',
    description='Framework for the estimation of questions latent traits from text.',
    author='Luca Benedetto',
    license='gpl-3.0',
    python_requires='>=3.7',
    install_requires=[
        'gensim==4.2.0',
        'nltk==3.7',
        'numpy==1.23.5',
        'pandas==1.5.2',
        'pyirt==0.3.4',
        'scikit-learn==1.1.3',
        'scipy==1.9.3',
        'textstat==0.7.3',
    ],
)
