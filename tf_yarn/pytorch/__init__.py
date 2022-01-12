import warnings

try:
    import torch
except ModuleNotFoundError:
    str = ("torch not found. "
           "You can install torch with 'pip install torch'"
           "or add it to the requirements.txt of your project.")
    warnings.warn(str)
