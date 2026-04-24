from setuptools import setup, find_packages

setup(name='asr_rl_pk',
      version='1.0.0',
      author='linser',
      author_email='',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Realization for the IPO RL algorithm',
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4"
      ],
      )
