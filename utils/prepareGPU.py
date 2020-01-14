def prepareGPU(ConfigProto):
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)