import os
import shutil
import traceback
import numpy as np
import matplotlib.pyplot as plt
from model import SiameseNetwork
from matplotlib.colors import hsv_to_rgb
from dataset import next_batch, load_data_from_directory, unison_shuffle


dataset_folder = './sign_dataset/'

embedding_size = 2

training_iterations = 5000
save_step = 100
batch_size = 64

image_height = 48
image_width = 48
image_channels = 3

amount_test_batches = 20
amount_test_images = batch_size * amount_test_batches

# Create the output directory.
shutil.rmtree('plots', ignore_errors=True)
os.makedirs('plots')
try:
    os.remove('train.log')
except OSError:
    pass

target_shape = (image_height, image_width, image_channels)

images, labels, labels_to_class_list = \
    load_data_from_directory(data_directory=dataset_folder,
                             target_shape=target_shape,
                             grayscale=False)
amount_classes = len(np.unique(labels))

images, labels = unison_shuffle(images, labels)
test_images = images[:amount_test_images]
test_labels = labels[:amount_test_images]
test_shape = (amount_test_batches, batch_size) + target_shape
test_images = test_images.reshape(test_shape)
del images
del labels

colours = []
for i in range(amount_classes):
    hsv = tuple(hsv_to_rgb([i / amount_classes, 1, 1]))
    colours.append(hsv)

marker_styles = ['.', 'o', '^', 's', 'p', '*', '+', 'x', 'D']

try:
    network = SiameseNetwork(input_image_shape=target_shape,
                             output_encoding_size=embedding_size)

    batch_generator = next_batch(batch_size,
                                 data_directory=dataset_folder,
                                 target_shape=target_shape, probability=0.1)

    for i, batch in enumerate(batch_generator):

        batch_left, batch_right, batch_similar = batch
        loss = network.optimise(batch_left, batch_right, batch_similar)

        print("loss: {}".format(loss))

        # Generate test embedding plots.
        if i % save_step == 0:
            print("testing.")

            embedding = np.array([network.inference(b) for b in test_images])
            embedding = embedding.reshape((-1, embedding_size))

            # plot result
            plt.figure(figsize=(16, 9))
            for j in range(amount_classes):
                xs = embedding[test_labels == j, 0].flatten()
                ys = embedding[test_labels == j, 1].flatten()
                marker_index = j % len(marker_styles)
                plt.plot(xs, ys,
                         marker_styles[marker_index],
                         c=colours[j],
                         alpha=0.8)

            plt.legend(labels_to_class_list)
            plt.savefig('plots/{}.jpg'.format(i))

    network.save()

except Exception as error:
    print("\nerror:\n", error)
    traceback.print_exc()
    network.close()

else:
    network.close()
    print("Done.")
