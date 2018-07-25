tf-skein
========

Train your TensorFlow [estimators][tf-estimators] on YARN using ~one~
only a few lines of code!

Installation
------------

```bash
$ pip install git+https://gitlab.criteois.com/s.lebedev/tf-skein.git
```

Usage
-----

Limitations
-----------

`tf-skein` uses [Miniconda][miniconda] for creating relocatable
Python environments. The package management, however, is done by
pip to allow for more flexibility. The downside to that is that
it is impossible to create an environment for an OS/architecture
different from the one the library is running on.

[miniconda]: https://conda.io/miniconda.html
[tf-estimators]: https://www.tensorflow.org/guide/estimators
