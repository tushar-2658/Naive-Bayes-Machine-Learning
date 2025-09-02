import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import openpyxl
from sklearn.model_selection import train_test_split

class Naive_Bayes:
    def __init__(this):
        this.classes = None
        this.class_priors = {}
        this.feature_stats = {} # for storing mean and standard dev for features
    
    def fit(this, X, y):
        this.classes = np.unique(y)
        n_samples = len(y)
        for class_label in this.classes:
            class_mask = (y == class_label)
            this.class_priors[class_label] = np.sum(class_mask) / n_samples

            #calc feature stats
            class_data = X[class_mask]
            this.feature_stats[class_label] = {
                'mean' : np.mean(class_data, axis = 0),
                'std' : np.std(class_data, axis = 0) + 1e-9 # adding offset to not divide by zero
            }

    def gauss(this, x, mean, std):
        exponent = -0.5 * ((x - mean) /std) ** 2
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)
    
    def predict_prob(this, X):
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(this.classes)))
        for i, class_label in enumerate(this.classes):
            class_prob = np.log(this.class_priors[class_label])
            mean = this.feature_stats[class_label]['mean']
            std = this.feature_stats[class_label]['std']

            for j in range(X.shape[1]):
                feature_prob = this.gauss(X[:, j], mean[j], std[j])
                class_prob += np.log(feature_prob + 1e-10)
            probabilities[:, i] = class_prob
        
        probabilities = np.exp(probabilities)
        probabilities = probabilities / np.sum(probabilities, axis = 1, keepdims = True)
        return probabilities
    def predict(this, X):
        probabilitues = this.predict_prob(X)
        return this.classes[np.argmax(probabilitues, axis = 1)]

    

def load_and_prep_data():
    df = pd.read_excel('Data.xlsx')
    print(f"Data loaded\nDataset shape : {df.shape}\ncolumns: {list(df.columns)}")
    if df.shape[1] == 31:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = [f"x{i + 1}" for i in range(30)]
    elif 'class' in df.columns or 'target' in df.columns:
        target_col = 'class' if 'class' in df.columns else 'target'
        X = df.drop(columns = [target_col]).values
        y = df[target_col].values
        feature_names = [col for col in df.columns if col != target_col]
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = [f"x{i + 1}" for i in range(X.shape[1])]
    print(f"feature shape: {X.shape}")
    print(f"target shape: {y.shape}")
    print(f"Classses: {np.unique(y)} (0 for milgnant 1 for benign)")
    print(f"class distribution: {np.bincount(y.astype(int))}")

    return X, y, feature_names


def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    y_pred_prob = classifier.predict_prob(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names = ['Malignant', 'Benign'])
    return accuracy, conf_matrix, class_report, y_pred, y_pred_prob

def main():
    X, y, feature_names = load_and_prep_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = 42, stratify = y
    )
    print(f"training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test[0].shape[0]} samples")
    print(f"training class distribution: {np.bincount(y_train)}")
    print(f"test class distribution: {np.bincount(y_test)}")

    print("Training Naive Bayes classifier ...")
    nb_classifier = Naive_Bayes()
    nb_classifier.fit(X_train, y_train)

    print("Testing classifier: ")
    accuracy, conf_matrix, class_report, y_pred, y_pred_prob = evaluate_classifier(
        nb_classifier, X_test, y_test
    )
    print(f"accuracy: {accuracy:.4f} ({accuracy*100:.2f})")
    print(f"\nConfusion Matrix: ")
    print(conf_matrix)
    print(f"\nClass Report: ")
    print(class_report)

    print("\n predicting for the given input vector: ")
    X_input = np.array([[13.0, 15.0, 85.0, 500.0, 0.1, 0.15, 0.1, 0.05, 0.2, 0.08, 0.5, 1.5, 4.0, 70.0, 0.01, 0.02,
                        0.02, 0.01, 0.015, 0.002, 14.0, 20.0, 90.0, 600.0, 0.2, 0.25, 0.2, 0.1, 0.3, 0.1]])
    prediction = nb_classifier.predict(X_input)
    prediction_prob = nb_classifier.predict_prob(X_input)

    print(f"input vector: {X_input[0]}")
    print(f"predicted class: {prediction[0]} ({'Benign' if prediction[0] == 1 else 'Malignant'})")
    print(f"prediction probabilities: ")
    print(f"P(Malignant | X) = {prediction_prob[0][0]:.4f}")
    print(f"P(Benign | X) = {prediction_prob[0][1]:.4f}")

    #plotting for visualization
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Blues", xticklabels = ['Malignant', 'Benign'], yticklabels = ['Malignant', 'Benign'])
    plt.title('confusion matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')

    #plot 2 for feature importance
    plt.subplot(1,2,2)
    mean_diff = np.abs(nb_classifier.feature_stats[0]['mean'] - nb_classifier.feature_stats[1]['mean'])
    top_features_idx = np.argsort(mean_diff)[-10:] #top 10 features that differentiates classes
    plt.barh(range(10), mean_diff[top_features_idx])
    plt.yticks(range(10), [feature_names[i][:15] + '...' if len(feature_names[i]) > 15 else feature_names[i] for i in top_features_idx])
    plt.xlabel('Mean difference')
    plt.title('Top 10 discriminative features')
    plt.tight_layout()

    plt.show()

    return nb_classifier, accuracy, prediction, prediction_prob



if __name__  == "__main__":
    classifier, accuracy, prediction, prediction_prob = main()
    print("Summary Results")
    print(f"Final accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Prediction for given input: {prediction[0]} ({'Benign' if prediction[0] == 1 else 'Malignant'})")
    print(f"Confidence: {max(prediction_prob[0]):.4f}")