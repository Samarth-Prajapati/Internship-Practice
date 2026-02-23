import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def load_dataset(path):
    """
    Loading Dataset

    Parameters
    ----------
    path - dataset path

    Returns - dataframe
    -------
    """

    data = pd.read_csv(path)

    print(f"Loading Dataframe = \n\n{data.head()}")
    print(f"Dataframe Shape = {data.shape}")

    return data

def check_dataset(data):
    """
    Checking Dataset

    Parameters
    ----------
    data - dataframe

    Returns - None
    -------
    """

    print("Dataset Info = \n")
    data.info()

    print(f"\nDataset Description = \n\n{data.describe()}")
    print(f"\nNumber of Duplicates in Dataset = {data.duplicated().sum()}")

def eda(data):
    """
    Performing EDA

    Parameters
    ----------
    data - dataframe

    Returns - None
    -------
    """

    print("EDA", end = " ")

    corr = data.corr()
    sns.heatmap(corr, annot = True)
    plt.show()

    for i,col in enumerate(data.columns):
        plt.subplot(5,4,i+1)
        sns.boxplot(y = data[col])
        plt.title(col)
    plt.show()

    plt.figure(figsize = (6, 4))
    sns.histplot(data["Insulin"], kde = True)
    plt.title("Insulin")
    plt.show()

    print("Analysis Successful.")

def handle_outliers(data):
    """
    Removing Outliers
    Parameters
    ----------
    data - dataframe

    Returns - None
    -------
    """

    data["Insulin"] = np.log1p(data["Insulin"])
    data["BloodPressure"] = np.log1p(data["BloodPressure"])
    data["DiabetesPedigreeFunction"] = np.log(data["DiabetesPedigreeFunction"])

    print("Handled Outliers using np.log function.")

    for i,col in enumerate(data.columns):
        plt.subplot(5, 4, i + 1)
        sns.boxplot(y = data[col])
        plt.title(col)
    plt.show()

def splitting_and_scaling(data):
    """
    Splitting & Scaling
    Parameters
    ----------
    data - dataframe

    Returns - xtrain, xtest, ytrain, ytest
    -------
    """

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 42)
    print(f"X_train Shape = {xtrain.shape}, X_test Shape = {xtest.shape}, y_train Shape = {ytrain.shape}, y_test Shape = {ytest.shape}")

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    return xtrain, xtest, ytrain, ytest

def training_model(xtrain, ytrain):
    """
    Training Model
    Parameters
    ----------
    xtrain - x_train
    ytrain - y_train

    Returns - logistic regression model
    -------
    """

    lr = LogisticRegression(class_weight = "balanced", random_state = 42)

    lr.fit(xtrain, ytrain)
    print("Logistic Regression Model Training Successful.")

    return lr

def check_model(lr_model, xtest, ytest):
    """
    Checking Model
    Parameters
    ----------
    lr_model : model
    xtest
    ytest

    Returns - ypred, accuracy, confusion_matrix
    -------
    """

    ypred = lr_model.predict(xtest)

    accuracy = accuracy_score(ytest, ypred)
    print(f"Accuracy = {accuracy * 100 : .2f}%\n")

    cm = confusion_matrix(ytest, ypred)
    print(f"Confusion Matrix = \n{cm}")

    cr = classification_report(ytest, ypred)
    print(f"\nClassification Report = \n{cr}")

    return ypred


if __name__ == "__main__":
    print("\n============================================= Logistic Regression =============================================\n")

    print("\n================================================= Load Dataset ================================================\n")
    df = load_dataset("diabetes.csv")

    print("\n================================================ Check Dataset ================================================\n")
    check_dataset(df)

    print("\n===================================================== EDA =====================================================\n")
    eda(df)

    print("\n=============================================== Remove Outliers ===============================================\n")
    handle_outliers(df)

    print("\n============================================= Splitting & Scaling =============================================\n")
    X_train, X_test, y_train, y_test = splitting_and_scaling(df)

    print("\n================================================ Training Model ===============================================\n")
    model = training_model(X_train, y_train)

    print("\n================================================ Checking Model ===============================================\n")
    y_predict = check_model(model, X_test, y_test)