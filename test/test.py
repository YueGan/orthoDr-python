import sys
import numpy as np
import test.cpp_exports as aw
from hypothesis import given, strategies as st 

# Local test
import python.silverman




if __name__ == "__main__":
    
    thing = np.arange(16, dtype=np.int32).reshape((4,4))
    mat_thing = aw.intmat(thing)
    print(mat_thing)

    # silverman.py
    print(python.silverman.silverman(1, 300))