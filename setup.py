#!/usr/bin/env python
# Copyright 2018 Criteo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "Distributed TensorFlow on a YARN cluster."

try:
    LONG_DESCRIPTION = open(os.path.join(here, "README")).read()
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
    version=_check_add_criteo_environment("0.1.11"),
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
    url="https://github.com/criteo/tf-yarn"
)
