import importlib.util as u
import numpy
print("NUMPY", numpy.__version__)
for m in ["scipy", "librosa", "soundfile", "matplotlib", "numba", "audioread", "soxr", "keras", "tensorflow"]:
    print(m, u.find_spec(m) is not None)
