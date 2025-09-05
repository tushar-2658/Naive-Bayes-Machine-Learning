import numpy as np
import pandas as pd
import openpyxl
import random


# commented code in python are for debugging purposes, commented english are explainations

class Naive_Bayes: # class for Naive Bayes
    def __init__(this):
        this.classes = None
        this.class_priors = {}
        this.feature_stats = {} # for storing mean and standard dev for features
    
    def fit(this, X, y): #fit function to train model
        this.classes = np.unique(y)
        n_samples = len(y) # total number of train test data
        X = np.array(X)
        y = np.array(y)
        for class_label in this.classes:
            class_mask = (y == class_label) # class mask is 0 or 1 depending on label
            this.class_priors[class_label] = np.sum(class_mask) / n_samples # calculating priors for each class

            #calc feature stats
            class_data = X[class_mask]
            # calc mean and standard dev for classes
            this.feature_stats[class_label] = {
                'mean' : np.mean(class_data, axis = 0),
                'std' : np.std(class_data, axis = 0) + 1e-9 # adding offset to not divide by zero
            }

    def gauss(this, x, mean, std): # calc gauss for given tests
        exponent = -0.5 * ((x - mean) /std) ** 2
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)
    
    def predict_prob(this, X): # predicts proba for given X vector
        X = np.array(X)
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(this.classes))) #initialize prob array with 0s
        for i, class_label in enumerate(this.classes): #iterating through classes
            class_prob = np.log(this.class_priors[class_label]) # calculating class prob for each class i and log(x) + log(y) = log(x * y)
            mean = this.feature_stats[class_label]['mean'] #mean and std dev for each class
            std = this.feature_stats[class_label]['std']

            for j in range(X.shape[1]):
                feature_prob = this.gauss(X[:, j], mean[j], std[j]) #feeding jth column, mean and std to gauss
                class_prob += np.log(feature_prob + 1e-10) # log(x) + log(y) = log(x * y)
            probabilities[:, i] = class_prob # storing class probabilities
        
        probabilities = np.exp(probabilities) # rollback log value to get actual value
        probabilities = probabilities / np.sum(probabilities, axis = 1, keepdims = True)
        return probabilities
    
    def predict(this, X):
        probabilitues = this.predict_prob(X)
        return this.classes[np.argmax(probabilitues, axis = 1)] # return index of greater probability

    

def load_and_prep_data():
    df = pd.read_excel('Data.xlsx')
#    print(f"Data loaded\nDataset shape : {df.shape}\ncolumns: {list(df.columns)}")
    X = df.iloc[:, :-1].values # select all rows and columns till the last
    y = df.iloc[:, -1].values # select all rows and last column
    #    feature_names = [f"x{i + 1}" for i in range(30)]
#    print(f"feature shape: {X.shape}")
#    print(f"target shape: {y.shape}")
#    print(f"Classses: {np.unique(y)} (0 for milgnant 1 for benign)")
#    print(f"class distribution: {np.bincount(y.astype(int))}")

    return X, y

def data_split(X, y, cut_off = 0.25, random_seed = 50):
    # change random seed value to generate different random numbers on each run
    # cutoff is the percentage of data that goes to test
    random.seed(random_seed)
    np.random.seed(random_seed)
    idx = list(range(len(X))) #indexes to shuffle
    random.shuffle(idx)
    split_point = int(569 * (1.0 - cut_off)) # 569 samples total
    train_idx = idx[:split_point] # all data till the split point
    test_idx = idx[split_point:] # all data including and after split point
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test

def calc_accuracy(y_true, y_pred):
    correct = sum(1 for res, pred in zip(y_true, y_pred) if res == pred) # number of correct predictions
    tot = len(y_true) # total number of samples
    return correct / tot

def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test) # gives the class number which is most likely correct
    y_pred_prob = classifier.predict_prob(X_test) # gives probability of class number which is most likely correct
    accuracy = calc_accuracy(y_test, y_pred)
    return accuracy, y_pred, y_pred_prob

def user_input(): # taking user input of 30 values
    values = []
    for i in range(30):
        values.append(float(input(f"Value {i + 1}: ")))
    return np.array(values)

def main():
    X, y= load_and_prep_data()
#    print(X)
    X_train, X_test, y_train, y_test = data_split(
        X, y
    )
#    print(f"training class distribution: {np.bincount(y_train)}")
#    print(f"test class distribution: {np.bincount(y_test)}")

    print("Training Naive Bayes classifier ...")
    nb_classifier = Naive_Bayes()
    nb_classifier.fit(X_train, y_train)

    print("Testing classifier: ")
    accuracy, y_pred, y_pred_prob = evaluate_classifier(
        nb_classifier, X_test, y_test
    )
    print(f"accuracy: {accuracy:.4f} ({accuracy*100:.2f})")

    print("\n predicting for the given input vector: ")
    X_input = np.array([[13.0, 15.0, 85.0, 500.0, 0.1, 0.15, 0.1, 0.05, 0.2, 0.08, 0.5, 1.5, 4.0, 70.0, 0.01, 0.02,
                        0.02, 0.01, 0.015, 0.002, 14.0, 20.0, 90.0, 600.0, 0.2, 0.25, 0.2, 0.1, 0.3, 0.1]])
    yes = input("would you like to input class manually? (By choosing N, it will give the result for provided test in the assignment) Y/N : ")
    if yes == 'Y':
        val = int(input("Enter the number of datasets you want to test: "))
        for i in range(val):
            X_input = np.array([user_input()])
            prediction = nb_classifier.predict(X_input) # gives class that the sample is most likely in
            prediction_prob = nb_classifier.predict_prob(X_input) # gives probability of the sample to be in the output class
            print(f"Prediction for given X vector: {'Benign' if prediction == 1 else 'Malignant'}  and the probabilities underlying are {prediction_prob[0][0]} for Benign and {prediction_prob[0][1]} for Malignant")
    else :
        prediction = nb_classifier.predict(X_input) # gives class that the sample is most likely in
        prediction_prob = nb_classifier.predict_prob(X_input) # gives probability of the sample to be in the output class
        print(f"Prediction for given X vector: {'Benign' if prediction == 1 else 'Malignant'}  and the probabilities underlying are {prediction_prob[0][0]} for Benign and {prediction_prob[0][1]} for Malignant")
    return nb_classifier, accuracy, prediction, prediction_prob

if __name__  == "__main__":
    classifier, accuracy, prediction, prediction_prob = main()
    print("Summary Results")
    print(f"Final accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)") # prints accuracy to 2 decimal place
  #  print(f"Prediction for given X vector: {'Benign' if prediction == 1 else 'Malignant'}  and the probabilities underlying are {prediction_prob[0][0]} for Benign and {prediction_prob[0][1]} for Malignant")