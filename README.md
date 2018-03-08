# Siamese Network Tensorflow

Siamese network is a neural network that contain two or more identical subnetwork. The purpose of this network is to find the similarity or comparing the relationship between two comparable things. Unlike classification task that uses cross entropy as the loss function, siamese network usually uses contrastive loss or triplet loss.

This project follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the output of the shared network and by optimizing the contrastive loss (see paper for more details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

## Run
Train the model
```bash
git clone https://github.com/ardiya/siamesenetwork-tensorflow
python train.py
```

Tensorboard Visualization(After training)
```bash
tensorboard --logdir=train.log
```

## Image retrieval
Image retrieval uses the trained model to extract the features and get the most similar image using cosine similarity.
[See here](https://github.com/ardiya/siamesenetwork-tensorflow/blob/master/Similar%20image%20retrieval.ipynb "See the code here")
