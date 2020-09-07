import setuptools

with open("readme.rst", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml_nn',
    version='0.4.1',
    author="Varuna Jayasiri, Nipun Wijerathne",
    author_email="vpjayasiri@gmail.com, hnipun@gmail.com",
    description="A collection of PyTorch implementations of neural network architectures and layers.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/lab-ml/labml_nn",
    project_urls={
        'Documentation': 'https://lab-ml.com/'
    },
    packages=setuptools.find_packages(exclude=('test',
                                               'test.*')),
    install_requires=['labml',
                      'labml_helpers',
                      'torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine learning',
)
