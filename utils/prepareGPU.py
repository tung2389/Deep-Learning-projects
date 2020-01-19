from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def prepareGPU():
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.71
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    return session

