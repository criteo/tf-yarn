import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "Distributed TensorFlow on a YARN cluster"

try:
    LONG_DESCRIPTION = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except Exception:
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


def get_tensorflow_version() -> str:
    version_gpu = ">=1.12.0,<2.0"
    version_cpu = ">=1.12.2,<2.0"
    if "BUILD_GPU" in os.environ:
        if "CRITEO_ENV" in os.environ:
            return f"tensorflow_gpu{version_gpu}.criteo"
        else:
            print("Warning: Public GPU build of tensorflow has not been tested with tf-yarn.")
            return "tensorflow_gpu"
    else:
        return f"tensorflow{version_cpu}"


def get_tfyarn_suffix() -> str:
    return "_gpu" if "BUILD_GPU" in os.environ else ""


setuptools.setup(
    name="tf_yarn" + get_tfyarn_suffix(),
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"tf_yarn": ["default.log.conf"]},
    version=_check_add_criteo_environment("0.4.18"),
    install_requires=REQUIREMENTS + [get_tensorflow_version()],
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
