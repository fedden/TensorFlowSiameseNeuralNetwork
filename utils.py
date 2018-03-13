import os
import torch
import shutil
import matplotlib
matplotlib.use('Agg')

import numpy as np
from PIL import Image as pil_image
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb


def plot_embedding(amount_classes, 
                   labels_to_class_list, 
                   embedding,
                   test_labels, 
                   plot_number, 
                   step):
    
    # Get colours and styles for plotting.
    marker_styles, colours = get_markers_and_colours(amount_classes)
    
    plt.figure(figsize=(16, 9))
    for j in range(amount_classes):
        a = plot_number * 2
        b = plot_number * 2 + 1
        xs = embedding[test_labels == j, a].flatten()
        ys = embedding[test_labels == j, b].flatten()
        marker_index = j % len(marker_styles)
        plt.plot(xs, ys,
                 marker_styles[marker_index],
                 c=colours[j],
                 alpha=0.8)

    plt.legend(labels_to_class_list)
    plt.savefig('plots/{}/{}.jpg'.format(plot_number, step))


def plot_loss(losses, steps):
    plt.figure(figsize=(16, 9))
    plt.plot(steps, losses)
    plt.savefig('plots/loss_history.jpg')
    plt.close()
    

def setup_working_directory(amount_plots, 
                            plot_directory, 
                            save_directory):
    
    # Wipe and create the output directory.
    shutil.rmtree(plot_directory, ignore_errors=True)
    os.makedirs(plot_directory)
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    for i in range(amount_plots):
        os.makedirs(os.path.join(plot_directory, str(i))
    
    
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_markers_and_colours(amount_classes):
    colours = []
    for i in range(amount_classes):
        hsv = tuple(hsv_to_rgb([i / amount_classes, 1, 1]))
        colours.append(hsv)

    marker_styles = ['.', 'o', '^', 's', 'p', '*', '+', 'x', 'D']
        
    while len(marker_styles) < amount_classes:
        marker_styles += marker_styles
    
    return marker_styles[:amount_classes], colours


def img_to_array(img, data_format='channels_last'):

    x = np.asarray(img, dtype=np.float32)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None, interpolation='nearest'):

    img = pil_image.open(path)

    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            img = img.resize(width_height_tuple, pil_image.BICUBIC)
    return img
