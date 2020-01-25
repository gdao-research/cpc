import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from cpc.data_handler import DataHandler
from cpc.model import CPCModel
np.random.seed(113)
tf.random.set_seed(113)


class DataTransform(DataHandler):
    def __init__(self, cpc_model, batch_size, terms, predict_terms=1, image_size=64, color=False, rescale=True, aug=True, is_training=True, method='cpc'):
        self.cpc = cpc_model
        super(DataTransform, self).__init__(batch_size, terms, predict_terms, image_size, color, rescale, aug, is_training, method)

    def __next__(self):
        x, y = self.benchmark_batch()
        z = self.cpc.get_encoding(x)
        return z, y


if __name__ == '__main__':
    ### TSNE CPC
    cpc = CPCModel(code_size=128, predict_terms=4, terms=4, units=256, image_size=64, channels=3)
    latest = tf.train.latest_checkpoint('weights/')
    cpc.load_weights(latest)
    df_test = DataTransform(cpc, 10000, 4, predict_terms=4, image_size=64, color=True, rescale=True, aug=True, is_training=False, method='cpc')
    x, y = next(df_test)
    idxs = []
    for i in range(10):
        idxs.append(np.where(y == i)[0])
    x_embedded = TSNE(verbose=1).fit_transform(x)
    fig = plt.figure(figsize=(20, 20))
    for i, idx in enumerate(idxs):
        plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], alpha=0.3, label=str(i))
    plt.legend()
    plt.savefig('images/tsne_cpc.png', dpi=100)

    ### TSNE raw pixel
    dh_test = DataHandler(10000, 4, predict_terms=4, image_size=64, color=True, rescale=True, aug=True, is_training=False, method='benchmark')
    x, y = next(df_test)
    x = x.reshape((x.shape[0], -1))
    idxs = []
    for i in range(10):
        idxs.append(np.where(y == i)[0])
    x_embedded = TSNE(verbose=1).fit_transform(x)
    fig = plt.figure(figsize=(20, 20))
    for i, idx in enumerate(idxs):
        plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], alpha=0.3, label=str(i))
    plt.legend()
    plt.savefig('images/tsne_pixel.png', dpi=100)
