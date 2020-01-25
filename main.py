import tensorflow as tf
from cpc.data_handler import DataHandler
from cpc.model import CPCModel


if __name__ == '__main__':
    dh_train = DataHandler(64, 4, predict_terms=4, image_size=64, color=True, rescale=True, aug=True, is_training=True, method='cpc')
    dh_test = DataHandler(64, 4, predict_terms=4, image_size=64, color=True, rescale=True, aug=True, is_training=False, method='cpc')
    accuracy_metric_train = tf.keras.metrics.BinaryAccuracy()
    loss_metric_train = tf.keras.metrics.BinaryCrossentropy()
    accuracy_metric_test = tf.keras.metrics.BinaryAccuracy()
    loss_metric_test = tf.keras.metrics.BinaryCrossentropy()
    cpc = CPCModel(code_size=128, predict_terms=4, terms=4, units=256, image_size=64, channels=3)
    optim = tf.keras.optimizers.Adam(1e-3)
    cb = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4),
          tf.keras.callbacks.ModelCheckpoint('weights/weights.{epoch:02d}-{val_binary_accuracy:.2f}.cpkt',
                                             monitor='val_binary_accuracy', save_best_only=True, save_weights_only=True),
          tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=3),
          tf.keras.callbacks.TensorBoard()]
    cpc.compile(optimizer=optim, loss='binary_crossentropy', metrics=['binary_accuracy'])
    cpc.fit(x=dh_train, epochs=20, validation_data=dh_test, steps_per_epoch=60000//64, validation_steps=10000//64, callbacks=cb)
