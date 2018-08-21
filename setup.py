from setuptools import setup, find_packages

setup(
    name="tf_yarn",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
      "tensorflow==1.10",
      "dill==0.2.8",
      "skein"
    ],
    python_requires=">=3.6"
)

