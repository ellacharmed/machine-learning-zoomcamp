import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

TF_CPP_MIN_LOG_LEVEL=2

print(f"Tensor Flow Version: {tf.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
