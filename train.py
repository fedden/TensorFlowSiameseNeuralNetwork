import numpy as np
import matplotlib.pyplot as plt
from model import SiameseNetwork
from dataset import next_batch, load_data_from_directory, unison_shuffle


training_iterations = 5000
save_step = 100
batch_size = 200

image_height = 48
image_width = 48
image_channels = 3

target_shape = (image_height, image_width, image_channels)

images, labels = load_data_from_directory(data_directory='./signs/',
                                          target_shape=target_shape,
                                          grayscale=False)
amount_classes = len(np.unique(labels))
images, labels = unison_shuffle(images, labels)

split = int(len(images) * 0.5)
test_images = images[split:]
test_labels = labels[split:]
del images
del labels

c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
c = c[:amount_classes]

try:
    network = SiameseNetwork(input_image_shape=target_shape,
                             output_encoding_size=2)

    # train iter
    for i, batch in enumerate(next_batch(batch_size, target_shape=target_shape)):

        batch_left, batch_right, batch_similar = batch
        loss = network.optimise(batch_left, batch_right, batch_similar)

        print("loss: {}".format(loss))

        if i % save_step == 0:
            # generate test
            embedding = network.inference(test_images)

            # plot result
            plt.figure(figsize=(16, 9))
            for j in range(amount_classes):
                xs = embedding[test_labels == j, 0].flatten()
                ys = embedding[test_labels == j, 1].flatten()
                plt.plot(xs, ys, '.', c=c[j], alpha=0.8)

            plt.legend(['0', '1', '2', '3'])
            plt.savefig('img/{}.jpg'.format(i))

    network.save()

except:
    network.close()

else:
    network.close()
    print("Done.")
