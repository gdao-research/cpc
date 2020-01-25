import tensorflow as tf
from cpc.data_handler import DataHandler
from cpc.model import CPCModel


class DataTransform(DataHandler):
    def __init__(self, cpc_model, batch_size, terms, predict_terms=1, image_size=64, color=False, rescale=True, aug=True, is_training=True, method='cpc'):
        self.cpc = cpc_model
        super(DataTransform, self).__init__(batch_size, terms, predict_terms, image_size, color, rescale, aug, is_training, method)

    def __next__(self):
        x, y = self.benchmark_batch()
        z = self.cpc.get_encoding(x)
        return z, y


if __name__ == '__main__':
    cpc = CPCModel(code_size=128, predict_terms=4, terms=4, units=256, image_size=64, channels=3)
    latest = tf.train.latest_checkpoint('weights/')
    cpc.load_weights(latest)
    df_train = DataTransform(cpc, 64, 4, predict_terms=4, image_size=64, color=True, rescale=True, aug=True, is_training=True, method='cpc')
    df_test = DataTransform(cpc, 64, 4, predict_terms=4, image_size=64, color=True, rescale=True, aug=True, is_training=False, method='cpc')
    linear_model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='softmax')])
    linear_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    linear_model.fit(x=df_train, epochs=10, validation_data=df_test, steps_per_epoch=60000//64, validation_steps=10000//64)
