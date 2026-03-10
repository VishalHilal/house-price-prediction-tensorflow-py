import tensorflow as tf
import numpy as np

x = np.array([1,2,3,4,5], dtype=float)
y = np.array([4,6,8,10,12], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

model.compile(
        optimizer='sgd',
        loss='mean_squared_error'
        )

model.fit(x,y, epochs=500, verbose=0)

print(model.predict(np.array([10.0])))
