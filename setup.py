import setuptools

with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml-nn',
    version='0.4.105',
    author="Varuna Jayasiri, Nipun Wijerathne",
    author_email="vpjayasiri@gmail.com, hnipun@gmail.com",
    description="🧠 Implementations/tutorials of deep learning papers with side-by-side notes; including transformers (original, xl, switch, feedback, vit), optimizers(adam, radam, adabelief), gans(dcgan, cyclegan, stylegan2), reinforcement learning (ppo, dqn), capsnet, distillation, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/labmlai/annotated_deep_learning_paper_implementations",
    project_urls={
        'Documentation': 'https://nn.labml.ai'
    },
    packages=setuptools.find_packages(exclude=('labml', 'labml.*',
                                               'labml_samples', 'labml_samples.*',
                                               'labml_helpers', 'labml_helpers.*',
                                               'test',
                                               'test.*')),
    install_requires=['labml>=0.4.110',
                      'labml-helpers>=0.4.77',
                      'torch',
                      'einops',
                      'numpy'],
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
