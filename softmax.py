import pandas as pd
import numpy as np

# Constants
FEATURE_COLUMNS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
TARGET_COLUMNS = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
SPECIES_MAPPING = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

def load_and_preprocess_data(filename='Iris.csv'):
    """Load the iris dataset and create one-hot encoded target variables."""
    data = pd.read_csv(filename)
    
    # Create one-hot encoded columns for species
    for species in TARGET_COLUMNS:
        data[species] = (data['Species'] == species).astype(int)
    
    return data

def train_test_split(data, test_ratio=0.2):
    """Split data into train and test sets using every 5th sample for testing."""
    train_data = []
    test_data = []
    
    for i in range(len(data)):
        if i % (1/test_ratio) == 0: 
            test_data.append(data.iloc[i])
        else:
            train_data.append(data.iloc[i])
    
    train_df = pd.DataFrame(train_data).drop(columns='Id').reset_index(drop=True)
    test_df = pd.DataFrame(test_data).drop(columns='Id').reset_index(drop=True)
    
    return train_df, test_df

def prepare_features_and_targets(df):
    """Separate features and targets from the dataframe."""
    features = df[FEATURE_COLUMNS]
    targets = df[TARGET_COLUMNS]
    return features, targets

def softmax(scores):
    """Apply softmax function to convert scores to probabilities."""
    exp_scores = np.exp(scores)
    score_sums = np.array(np.sum(exp_scores, axis=1))
    return exp_scores / score_sums.reshape(-1,1)

def predict(weights, data):
    """Make predictions using the trained weights."""
    scores = np.matmul(data, np.transpose(weights))
    probabilities = softmax(scores)
    return probabilities

def train(data, labels, learning_rate=0.01, epochs=1000):
    """Train softmax regression model using gradient descent."""
    n_features = data.shape[1]
    n_classes = labels.shape[1]
    
    # Initialize weights with small random values instead of ones
    weights = np.random.normal(0, 0.01, (n_classes, n_features))
    
    for epoch in range(epochs):
        probabilities = predict(weights, data)
        gradient = np.dot((probabilities - labels.values).T, data) / len(labels)
        weights -= learning_rate * gradient
        
        # Optional: print loss every 100 epochs for monitoring
        if epoch % 100 == 0:
            loss = -np.mean(np.sum(labels.values * np.log(probabilities + 1e-15), axis=1))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights

def accuracy(probabilities, target):
    """Calculate accuracy by comparing predicted classes with true classes."""
    predicted_classes = np.argmax(probabilities, axis=1)
    true_classes = np.argmax(target.values, axis=1)
    correct = np.sum(predicted_classes == true_classes)
    return correct / len(target)


def main():
    """Main function to run the softmax regression."""
    # Load and preprocess data
    data = load_and_preprocess_data('Iris.csv')
    train_data, test_data = train_test_split(data)
    
    # Prepare features and targets
    train_features, train_targets = prepare_features_and_targets(train_data)
    test_features, test_targets = prepare_features_and_targets(test_data)
    
    print(f"Training data shape: {train_features.shape}")
    print(f"Test data shape: {test_features.shape}")
    
    # Train the model
    trained_weights = train(train_features, train_targets, learning_rate=0.01, epochs=1000)
    
    # Evaluate on training data
    train_probabilities = predict(trained_weights, train_features)
    train_accuracy = accuracy(train_probabilities, train_targets)
    
    # Evaluate on test data
    test_probabilities = predict(trained_weights, test_features)
    test_accuracy = accuracy(test_probabilities, test_targets)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()

    