import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def classify_from_embeddings(model, 
                             train_images, 
                             train_labels, 
                             test_images, 
                             test_labels,
                             k=5, 
                             distance_metric='mahalanobis',
                             distance_weighting='distance'):
    
    # Create training embeddings.
    train_embeddings = np.array([model.inference(b) for b in train_images])
    train_embeddings = train_embeddings.reshape((-1, model.embedding_size))
    
    # Create testing embeddings.
    test_embeddings = np.array([model.inference(b) for b in testing_images])
    test_embeddings = test_embeddings.reshape((-1, model.embedding_size))
    
    # Train kNN.
    classifier = KNeighborsClassifier(n_neighbors=k, 
                                      weights=distance_weighting,
                                      algorithm='auto',
                                      metric=distance_metric, 
                                      n_jobs=-1)
    classifier.fit(train_embeddings, train_labels)
    
    # Get predictions.
    test_predictions = classifier.predict(test_embeddings)
    
    # Return accuracy of kNN.
    accuracy = classifier.score(test_labels, test_predictions)
    return accuracy


