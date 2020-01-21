import numpy as np
import matplotlib.pyplot as plt
from cpc.data_handler import DataHandler

dh = DataHandler(64, 4, 4, color=True, rescale=True)
# images, labels = dh.cpc_batch()
# print(images[0].shape, images[1].shape, labels.shape)
# x = np.concatenate(images, axis=1)
# print(x.shape)
# print(labels.flatten())
# fig = plt.figure()
# for i in range(8):
#     fig.add_subplot(1, 8, i+1)
#     plt.imshow(x[-1, i, ...])
#     plt.axis('off')
# plt.savefig('sample.png')
# plt.close()
# # plt.show()

images, labels = dh.benchmark_batch()
