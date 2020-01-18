  keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),

  keras.layers.Dropout(0.5),
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(5, activation='softmax')

  epochs = 60