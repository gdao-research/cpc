import tensorflow as tf


class CPCModel(tf.keras.Model):
    def __init__(self, code_size, predict_terms, terms=4, units=256, image_size=64, channels=3):
        super(CPCModel, self).__init__()
        self.code_size = code_size
        self.predict_terms = predict_terms
        self.terms = terms
        self.units = units
        self.image_size = image_size
        self.channels = channels

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.lrelu1 = tf.keras.layers.LeakyReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.lrelu2 = tf.keras.layers.LeakyReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.lrelu3 = tf.keras.layers.LeakyReLU()
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.lrelu4 = tf.keras.layers.LeakyReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.dense5 = tf.keras.layers.Dense(units=256, activation='linear')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.lrelu5 = tf.keras.layers.LeakyReLU()
        self.dense6 = tf.keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')

        self.gru = tf.keras.layers.GRU(units, return_sequences=False, name='ar_context')
        self.linear = tf.keras.layers.Dense(predict_terms*code_size, activation='linear')

    def get_encoding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)
        x = self.flatten(x)
        x = self.dense5(x)
        x = self.bn5(x)
        x = self.lrelu5(x)
        z = self.dense6(x)
        return z

    def get_context(self, x):
        z = self.get_encoding(x)
        z = tf.reshape(z, [-1, self.terms, self.code_size])
        c = self.gru(z)
        return c

    def get_prediction(self, x):
        c = self.get_context(x)
        z_hats = self.linear(c)
        z_hat = tf.reshape(z_hats, [-1, self.predict_terms, self.code_size])
        return z_hat

    def call(self, x):
        x_tm, x_tp = x
        x_tm = tf.reshape(x_tm, [-1, self.image_size, self.image_size, self.channels])
        x_tp = tf.reshape(x_tp, [-1, self.image_size, self.image_size, self.channels])
        z_hat = self.get_prediction(x_tm)
        z_tp = self.get_encoding(x_tp)
        z_tp = tf.reshape(z_tp, [-1, self.predict_terms, self.code_size])
        dot_prods = tf.reduce_mean(tf.reduce_mean(z_hat*z_tp, axis=-1), axis=-1, keepdims=True)
        probs = tf.sigmoid(dot_prods)
        return probs
