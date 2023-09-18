#!/usr/bin/env python
from setuptools import setup, find_packages

# Parse version number from pyglet/__init__.py:
with open('non_param_score_est/__init__.py') as f:
    info = {}
    for line in f:
        if line.startswith('version'):
            exec(line, info)
            break

extra_test = [
    'pytest>=4',
    'pytest-cov>=2',
]

extra_dev = [
    *extra_test,
]

extra_ci = [
    *extra_test,
    'python-coveralls',
]

setup_info = dict(
    name='non_param_score_est',
    version=info['version'],
    author='Krunoslav Lehman Pavasovic',
    author_email='krunolp@gmail.com',
    url='https://github.com/krunolp/score_estim',
    project_urls={
        'Source': 'https://github.com/krunolp/score_estim',
        'Tracker': 'https://github.com/krunolp/score_estim/issues',
    },
    description='Non parametric score function estimation library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: MacOS X',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',


    ],

    # Package info
    packages=find_packages(exclude=['tests', 'tests.*']),

    py_modules=['non_param_score_est'],

    # Add _ prefix to the names of temporary build dirs
    options={'build': {'build_base': '_build'}, },
    zip_safe=True,

    install_requires=[
        'jax>=0.4.1',
        'jaxlib>=0.4.1',
        'dm-haiku',
        'tensorflow_probability',
    ],

    extras_require={
        'dev': extra_dev,
        'test': extra_test,
    }
)

setup(**setup_info)
