import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "Distributed TensorFlow on a YARN cluster"

try:
    LONG_DESCRIPTION = open(os.path.join(here, "README.md")).read()
except IOError:
    LONG_DESCRIPTION = ""


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines()
                if (s.strip() and not s.startswith("#"))]


REQUIREMENTS = _read_reqs("requirements.txt")

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    "Environment :: Console",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Software Development :: Libraries"
]


def _check_add_criteo_environment(package_name):
    if "CRITEO_ENV" in os.environ:
        return package_name + "+criteo"

    return package_name


setup(
    name="tf_yarn",
    packages=["tf_yarn"],
    include_package_data=True,
    package_data={"tf_yarn": ["default.log.conf"]},
    version=_check_add_criteo_environment("0.4.1"),
    install_requires=REQUIREMENTS,
    tests_require=["pytest", "hadoop-test-cluster"],
    python_requires=">=3.6",

    maintainer="Criteo",
    maintainer_email="github@criteo.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    keywords="tensorflow yarn",
    url="https://github.com/criteo/tf-yarn",
    entry_points={'console_scripts': [
        'check_hadoop_env = tf_yarn.bin.check_hadoop_env:main',
    ]}
)
