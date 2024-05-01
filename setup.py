from setuptools import setup

setup(
    name='simple_worm',
    version='0.0.3',
    description='Python implementation of numerical method for visco-elastic rods.',
    author='Tom Ranner, Tom Ilett, Amelia Smyth, Saul Cooperman',
    url='https://github.com/uol-feps-soc-comp3931-2324-classroom/final-year-project-scopreon',
    packages=['simple_worm'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'fenics == 2019.1.0',
        'numpy >= 1.19, < 2',
        'scikit-learn >= 0.24'
    ],
    extras_require={
        'test': [
            'pytest'
        ],
        'inv': [
            'cyipopt >= 1.1, <= 1.2',
            'dolfin_adjoint @ git+https://github.com/dolfin-adjoint/pyadjoint.git@1c9c15c1fa2c1a470826143ce98b721ebd00facd',
            'torch >= 1.8, <= 1.9',
            'mkl-service >= 2.3, <= 2.4',
            'matplotlib >= 3.4',
            'tensorboard == 2.4.1',
        ]
    },
    python_requires=">=3.8, <3.10",
)
