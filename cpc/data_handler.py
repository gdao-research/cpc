import numpy as np
import cv2
import tensorflow as tf


class DataHandler:
    def __init__(self, batch_size, terms, predict_terms=1, image_size=64, color=False, rescale=True, aug=True, is_training=True, method='cpc'):
        self.batch_size = batch_size
        self.terms = terms
        self.predict_terms = predict_terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale
        self.aug = aug
        self.is_training = is_training
        self.method = method
        self.lena = cv2.imread('cpc/lena.jpg')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if self.is_training:
            self.x = x_train/255.0
            self.y = y_train
        else:
            self.x = x_test/255.0
            self.y = y_test
        self.idxs = []
        for i in range(10):
            y = y_train if self.is_training else y_test
            self.idxs.append(np.where(y == i)[0])
        self.n_samples = len(self.y)//terms if self.method == 'cpc' else len(self.y)
        self.shape = self.x.shape
        self.n_batches = self.n_samples//batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.cpc_batch() if self.method == 'cpc' else self.benchmark_batch()

    def __len__(self):
        return self.n_batches

    def cpc_batch(self):
        img_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        for bi in range(self.batch_size):
            seed = np.random.randint(10)
            sentence = np.arange(seed, seed + self.terms + self.predict_terms) % 10
            if bi < self.batch_size//2:
                num = np.arange(10)
                predicted = sentence[-self.predict_terms:]
                for i, p in enumerate(predicted):
                    predicted[i] = np.random.choice(num[num != p], 1)
                sentence[-self.predict_terms:] = predicted % 10
                sentence_labels[bi, :] = 0
            img_labels[bi, :] = sentence
        images = self.get_samples(img_labels).reshape((self.batch_size, self.terms+self.predict_terms, self.image_size, self.image_size, 3))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]
        idx = np.random.choice(self.batch_size, self.batch_size, replace=False)
        return [x_images[idx], y_images[idx]], sentence_labels[idx]

    def get_samples(self, img_labels):
        idx = []
        for label in img_labels.flatten():
            idx.append(np.random.choice(self.idxs[int(label)], 1)[0])
        img_batch = self.x[idx, :, :]
        if self.aug:
            img_batch = self._aug_batch(img_batch)
        return img_batch

    def _aug_batch(self, img_batch):
        if self.image_size != 28:
            resized = []
            for i in range(img_batch.shape[0]):
                resized.append(cv2.resize(img_batch[i], (self.image_size, self.image_size)))
            img_batch = np.stack(resized)
        img_batch = img_batch.reshape((img_batch.shape[0], 1, self.image_size, self.image_size))
        img_batch = np.concatenate([img_batch, img_batch, img_batch], axis=1)

        if self.color:
            img_batch[img_batch >= 0.5] = 1
            img_batch[img_batch < 0.5] = 0
            for i in range(img_batch.shape[0]):
                x_c = np.random.randint(0, self.lena.shape[0] - self.image_size)
                y_c = np.random.randint(0, self.lena.shape[1] - self.image_size)
                img = self.lena[x_c:x_c+self.image_size, y_c:y_c+self.image_size]
                img = np.array(img).transpose((2, 0, 1))/255.0
                for j in range(3):
                    img[j, :, :] = (img[j, :, :] + np.random.uniform(0, 1))/2.0
                img[img_batch[i, :, :, :] == 1] = 1 - img[img_batch[i, :, :, :] == 1]
                img_batch[i, :, :, :] = img

        if self.rescale:
            img_batch = img_batch * 2 - 1
        img_batch = img_batch.transpose((0, 2, 3, 1))
        return img_batch

    def benchmark_batch(self):
        idx = np.random.choice(len(self.x), self.batch_size, replace=False)
        img_batch = self.x[idx]
        label_batch = self.y[idx]
        if self.aug:
            img_batch = self._aug_batch(img_batch)
        label_batch = label_batch.reshape((-1, 1))
        return img_batch, label_batch
