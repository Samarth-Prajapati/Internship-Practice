import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_dataset(path):
    """
    Load Dataset
    Parameters
    ----------
    path - dataset path

    Returns - dataframe
    -------
    """

    df = pd.read_csv(path)
    print(f"DataFrame = \n\n{df.head()}")

    return df

def check_dataset(df):
    """
    Check Dataset
    Parameters
    ----------
    df - dataframe

    Returns - None
    -------
    """

    print("Dataset Info = \n")
    df.info()

    print(f"\nDataset Description = \n\n{df.describe()}\n")

    print(f"Count of Duplicate Values = {df.duplicated().sum()}")

def remove_duplicates(df):
    """
    Remove Duplicates
    Parameters
    ----------
    df - dataframe

    Returns - None
    -------
    """

    df.drop_duplicates(inplace = True)

    print(f"Duplicates Removed Successfully ( After Removing Duplicate Values, Duplicate Count = {df.duplicated().sum()} ).")

def perform_eda(df):
    """
    Perform EDA
    Parameters
    ----------
    df - dataframe

    Returns - None
    -------
    """

    corr = df.corr()
    sns.heatmap(corr, annot = True)
    plt.show()

    for i, col in enumerate(df.columns):
        plt.subplot(5, 4, i + 1)
        sns.boxplot(y = df[col])
        plt.title(col)
    plt.show()

    plt.figure(figsize = (6, 4))
    sns.histplot(df["skewness"], kde = True)
    plt.title("skewness")
    plt.show()

    print("EDA Performed Successfully.")

def handle_outliers(df):
    """
    Handle Outliers
    Parameters
    ----------
    df - dataframe

    Returns - None
    -------
    """

    cols = ["curtosis", "entropy"]

    print("Outliers Info = \n")
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"Feature ( {col} ) = {len(outliers)} outliers found.")

        df[col] = df[col].clip(lower = lower, upper = upper)

    print("\n\nHandled Outliers Successfully.")

    for i, col in enumerate(df.columns):
        plt.subplot(5, 4, i + 1)
        sns.boxplot(y = df[col])
        plt.title(col)
    plt.show()

def split_data(df):
    """
    Split Data
    Parameters
    ----------
    df - dataframe

    Returns - xtrain, xtest, ytrain, ytest
    -------
    """

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 42)

    print("Performed train_test_split Successfully.")

    return xtrain, xtest, ytrain, ytest

def train_model(xtrain, ytrain):
    """
    Train Model
    Parameters
    ----------
    xtrain
    ytrain

    Returns - model
    -------
    """

    tree = DecisionTreeClassifier(
        max_depth = 5,
        class_weight = "balanced",
        max_leaf_nodes = 2,
        random_state = 42
    )

    tree.fit(xtrain, ytrain)

    # plot_tree(tree)
    # plt.show()

    print("Model Trained Successfully.")

    return tree

def check_model(tree, xtest, ytest):
    """
    Check Model
    Parameters
    ----------
    tree
    xtest
    ytest

    Returns - None
    -------
    """

    ypred = tree.predict(xtest)

    acc = accuracy_score(ytest, ypred)
    print(f"Accuracy = {acc * 100 : 2f}\n")

    cm = confusion_matrix(ytest, ypred)
    print(f"Confusion Matrix = \n{cm}\n")

    cr = classification_report(ytest, ypred)
    print(f"Classification Report = \n\n{cr}")

if __name__ == "__main__":
    print("\n--------------------------------------------- DECISION TREE CLASSIFIER ---------------------------------------------\n")

    print("--------------------------------------------------- LOAD DATASET ---------------------------------------------------\n")
    data = load_dataset("BankNoteAuthentication.csv")

    print("\n--------------------------------------------------- CHECK DATASET --------------------------------------------------\n")
    check_dataset(data)

    print("\n------------------------------------------ REMOVE DUPLICATES FROM DATASET ------------------------------------------\n")
    remove_duplicates(data)

    print("\n-------------------------------------------------- PERFORMING EDA --------------------------------------------------\n")
    perform_eda(data)

    print("\n------------------------------------------------- HANDLING OUTLIERS ------------------------------------------------\n")
    handle_outliers(data)

    print("\n------------------------------------------------- TRAIN TEST SPLIT -------------------------------------------------\n")
    X_train, X_test, y_train, y_test = split_data(data)

    print("\n-------------------------------------------------- MODEL TRAINING --------------------------------------------------\n")
    model = train_model(X_train, y_train)

    print("\n------------------------------------------------- EVALUATING MODEL -------------------------------------------------\n")
    check_model(model, X_test, y_test)

    print("\n----------------------------------------------------- THE END ------------------------------------------------------\n")

