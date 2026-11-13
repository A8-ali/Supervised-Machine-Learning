import numpy as np
from collections import Counter

# Training Dataset
# Features: [sepal_length, sepal_width]
X_train = np.array([
    [5.1, 3.5],
    [4.9, 3.0],
    [5.0, 3.4],
    [5.4, 3.9],

    [6.0, 2.7],
    [6.2, 2.9],
    [5.9, 3.0],
    [6.1, 2.8],

    [7.0, 3.2],
    [6.8, 3.0],
    [7.2, 3.3],
    [6.9, 3.1]
])
y_train = np.array([
    "Setosa",
    "Setosa",
    "Setosa",
    "Setosa",

    "Versicolor",
    "Versicolor",
    "Versicolor",
    "Versicolor",

    "Virginica",
    "Virginica",
    "Virginica",
    "Virginica"
])

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, sample):

        # Calculate Euclidean distances
        distances = np.sqrt(
            np.sum((self.X_train - sample) ** 2, axis=1)
        )

        # Find indices of K nearest neighbors
        nearest_indices = np.argsort(distances)[:self.k]

        # Get labels of nearest neighbors
        nearest_labels = self.y_train[nearest_indices]

        # Majority voting
        prediction = Counter(nearest_labels).most_common(1)[0][0]

        return prediction

    def predict(self, X):

        return np.array([
            self.predict_one(sample)
            for sample in X
        ])

model = KNN(k=3)

model.fit(X_train, y_train)

X_test = np.array([
    [5.2, 3.6],
    [6.3, 2.6],
    [7.1, 3.2]
])

predictions = model.predict(X_test)


# Display results
for sample, prediction in zip(X_test, predictions):

    print(f"Input: {sample}")
    print(f"Predicted Species: {prediction}")
    print("-" * 30)
