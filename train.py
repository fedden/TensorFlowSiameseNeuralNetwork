import numpy as np
from model import SiameseModel
from dataset import next_batch, get_testing_batches
from utils import plot_embedding, plot_loss
from utils import get_markers_and_colours, setup_working_directory


dataset_directory = 'sign_dataset'
save_directory = 'saves'
plot_directory = 'plots'

embedding_size = 2
margin = 0.2
learning_rate = 0.01
batch_size = 64
training_iterations = 20000
save_step = 100
amount_plots = 1
amount_test_batches = 18

# Create folders to save models and plots.
setup_working_directory(amount_plots, 
                        plot_directory, 
                        save_directory)

# Produce H, W, C images that the model will convert.
image_height = 224
image_width = 224
image_channels = 3
image_shape = (image_height, image_width, image_channels)

# Get a testing set of batches for plotting.
test_images, test_labels, labels_to_class_list = \
    get_testing_batches(dataset_directory, 
                        image_shape,
                        amount_test_batches, 
                        batch_size)
    
amount_classes = len(np.unique(test_labels))

# The infinite online training batch generator.
batch_generator = next_batch(batch_size,
                             data_directory=dataset_directory,
                             target_shape=image_shape, 
                             probability=0.1)
# The siamese model.
model = SiameseModel(input_image_shape=image_shape,
                     output_size=embedding_size,
                     margin=margin,
                     learning_rate=learning_rate)

# Optimisation / visualisation process.
loss_history = []
step_history = []
for step, batch in enumerate(batch_generator):

    # Generate test embedding plots.
    if step % save_step == 0:
        print("\ntesting.")
        
        # Plot training loss.
        if len(loss_history) > 0: 
            plot_loss(loss_history, step_history)

        embedding = np.array([model.inference(b) for b in test_images])
        embedding = embedding.reshape((-1, embedding_size))
        
        # Create some plots from the embeddings.
        for plot_number in range(amount_plots):
            plot_embedding(amount_classes, 
                           labels_to_class_list, 
                           embedding,
                           test_labels, 
                           plot_number,
                           step)
        model.save(save_directory)
        
    # Training.
    batch_left, batch_right, batch_similar = batch
    loss = model.optimise_batch(*batch)
    
    # Append loss and step to history.
    loss_history.append(loss)
    step_history.append(step * batch_size)

    print("loss: {}".format(loss), end='\r')