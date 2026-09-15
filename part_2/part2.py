import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def plot_gallery(images, titles, h, w, n_row=3, n_col=4, filename='eigenfaces.png'):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    print("Fetching LFW dataset (this may take a moment)...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names

    print(f"Total dataset size: \n n_samples: {n_samples} \n n_features: {X.shape[1]} \n n_classes: {target_names.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    mean = np.mean(X_train, axis=0)
    X_train_centered = X_train - mean
    X_test_centered = X_test - mean

    n_components = 150
    print(f"Extracting the top {n_components} eigenfaces...")
    U, S, V = np.linalg.svd(X_train_centered, full_matrices=False)
    
    components = V[:n_components]
    eigenfaces = components.reshape((n_components, h, w))

    X_transformed = np.dot(X_train_centered, components.T)
    X_test_transformed = np.dot(X_test_centered, components.T)

    eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    explained_variance = (S ** 2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    ratio_cumsum = np.cumsum(explained_variance_ratio)

    plt.figure()
    plt.plot(np.arange(n_components), ratio_cumsum[:n_components])
    plt.title('Compactness')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.savefig('compactness_plot.png')
    print("Saved compactness_plot.png")

    print("Training Random Forest Classifier...")
    estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
    estimator.fit(X_transformed, y_train) 

    predictions = estimator.predict(X_test_transformed)
    correct = (predictions == y_test)
    total_test = len(X_test_transformed)

    print("\n--- Classification Performance ---")
    print("Total Testing:", total_test)
    print("Total Correct:", np.sum(correct))
    print(f"Accuracy: {np.sum(correct)/total_test:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

if __name__ == "__main__":
    main()