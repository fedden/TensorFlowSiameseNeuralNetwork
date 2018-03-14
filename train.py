import os
import json
import numpy as np
from model import SiameseModel
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from classify import classify_from_embeddings
from dataset import next_batch, get_image_label_batches
from utils import plot_embedding, plot_loss
from utils import get_markers_and_colours, setup_working_directory

# Directory structure and data sources.
generator_dataset_directory = 'sign_dataset'
test_dataset_directory = 'sign_dataset'
train_dataset_directory = 'sign_dataset'
save_directory = 'saves'
plot_directory = 'plots'

# Siamese model configuration.
embedding_size = 128
margin = 20.0 #2.0 # 0.2

# Siamese model training.
learning_rate = 0.01
batch_size = 64
training_iterations = 20000
save_step = 1000

# Plotting.
amount_plots = 4
amount_test_batches = 18

# Input image dims.
image_height = 224
image_width = 224
image_channels = 3

# kNN grid search.
grid_search_options = {
    'n_neighbors': [1, 5, 10, 15],
    'metric': ['euclidean', 'mahalanobis'],
    'weights': ['distance', 'uniform']
}

# Create folders to save models and plots.
setup_working_directory(amount_plots, 
                        plot_directory, 
                        save_directory)

# Produce H, W, C images that the model will convert.
image_shape = (image_height, image_width, image_channels)

## Get a testing set of batches for plotting.
#test_images, test_labels, _ = \
#    get_image_label_batches(test_dataset_directory, 
#                            image_shape,
#                            amount_test_batches, 
#                            batch_size)
#    
# Get a training set of batches for plotting.
#train_images, train_labels, labels_to_class_list = \
#    get_image_label_batches(train_dataset_directory, 
#                            image_shape,
#                            amount_test_batches, 
#                            batch_size)

# Get a training set of batches for plotting.
images, labels, labels_to_class_list = \
    get_image_label_batches(train_dataset_directory, 
                            image_shape,
                            amount_test_batches, 
                            batch_size)
amount_batches = amount_test_batches // 2
train_images = images[:amount_batches]

orig_shape = train_images.shape
s = len(train_images.reshape((-1,) + image_shape))
test_images = images[amount_batches:amount_batches*2]

train_labels = labels[:s]
test_labels = labels[s:s*2]
assert len(train_images) == len(test_images)
print(train_labels.shape, test_labels.shape)
assert len(train_labels) == len(test_labels)


# The infinite online training batch generator.
batch_generator = next_batch(batch_size,
                             data_directory=generator_dataset_directory,
                             target_shape=image_shape, 
                             probability=0.1)
# The siamese model.
model = SiameseModel(input_image_shape=image_shape,
                     output_size=embedding_size,
                     margin=margin,
                     learning_rate=learning_rate)

# Need this for plotting.
amount_classes = len(np.unique(train_labels))

# Optimisation / visualisation process.
loss_history = []
step_history = []
best_test_accuracy = 0.0
for step, batch in enumerate(batch_generator):

    # Generate test embedding plots.
    if step % save_step == 0:
        print("\ntesting.")
        
        # Plot training loss.
        if len(loss_history) > 0: 
            plot_loss(loss_history, step_history)

        # Create training embeddings.
        train_embeddings = np.array([model.inference(b) for b in train_images])
        train_embeddings = train_embeddings.reshape((-1, embedding_size))

        # Create testing embeddings.
        test_embeddings = np.array([model.inference(b) for b in test_images])
        test_embeddings = test_embeddings.reshape((-1, embedding_size))
        
        # Create some plots from the embeddings.
        for plot_number in range(amount_plots):
            plot_embedding(amount_classes, 
                           labels_to_class_list, 
                           test_embeddings,
                           test_labels, 
                           plot_number,
                           step)
        
        # Check accuracy of model using grid search.
        search_best_accuracy = 0.0
        for parameters in ParameterGrid(grid_search_options):

            # Need some extra stuff for mahalanobis distance.
            if parameters['metric'] == 'mahalanobis':
                parameters['metric_params'] = {
                    'V': np.cov(train_embeddings)
                }
                
            # Instanciate the knn.
            knn = KNeighborsClassifier(**parameters, 
                                       algorithm='brute',
                                       n_jobs=-1)
            knn.fit(train_embeddings, train_labels)
            
            # Get predictions.
            test_predictions = knn.predict(test_embeddings)

            # Testing accuracy of kNN.
            accuracy = accuracy_score(test_labels, test_predictions)
            
            if accuracy > search_best_accuracy:
                best_parameters = parameters
                search_best_accuracy = accuracy
                    
        # Save model and potentially kNN parameters.
        if search_best_accuracy > best_test_accuracy:
            best_test_accuracy = search_best_accuracy
            print('new best accuracy:', search_best_accuracy)
            print('params:', best_parameters)

            model.save(save_directory, is_best=True)
            save_path = os.path.join(save_directory, 'best_knn_params.json')
            with open(save_path, 'w') as out_file:
                json.dump(best_parameters, out_file)
        else:
            print('best effort accuracy:', search_best_accuracy)
            print('current best accuracy:', best_test_accuracy)
            model.save(save_directory, is_best=False)
            
        print()

    # Training.
    batch_left, batch_right, batch_similar = batch
    loss = model.optimise_batch(*batch)
    
    # Append loss and step to history.
    loss_history.append(loss)
    step_history.append(step * batch_size)

    print("loss: {}, step {}              "
          .format(loss, step * batch_size), end='\r')