import tensorflow as tf
from cpc.data_handler import DataHandler

dh_train = DataHandler(64, 4, 4, color=True, method='benchmark')
dh_test = DataHandler(64, 4, 4, color=True, is_training=False, method='benchmark')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=dh_train, epochs=10, steps_per_epoch=60000//64)
model.evaluate(x=dh_test, steps=10000//64)
