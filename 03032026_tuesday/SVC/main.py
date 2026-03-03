import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

seperator = f"\n\n{'--' * 70}\n\n"

class SupportVectorClassifier:
    """
    Support Vector Machine Classifier Implementation
    """

    def __init__(self, path):
        self.path = path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.3
        self.random_state = 42
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.model = SVC()
        self.y_pred = None

    def load_data(self):
        """
        Load data from csv file
        Returns - None
        -------
        """

        self.df = pd.read_csv(self.path)
        print(f"\nDataset = \n{self.df.head()}\n")

        print("Data Loaded Successfully.", end = seperator)

    def check_data(self):
        """
        Check if data is correct for model training
        Returns - None
        -------
        """

        print("Dataset Info = ")
        self.df.info()

        print(f"\nDataset Description = \n{self.df.describe()}\n")
        print(f"Total Duplicate Values = {self.df.duplicated().sum()}")
        print(f"Dataset Shape = {self.df.shape}")
        print(f"\nTotal Null Values = \n{self.df.isnull().sum()}\n")

        print("Data Checked Successfully.", end = seperator)

    def preprocess_data(self):
        """
        Preprocess data
        Returns - None
        -------
        """

        self.df.drop("user_id", axis = 1, inplace = True)
        print(f"Dataset = \n{self.df.head()}\n")

        print("Data Preprocessed Successfully.", end = seperator)

    def eda(self):
        """
        Perform EDA on data
        Returns - None
        -------
        """

        corr = self.df.corr(numeric_only = True)
        sns.heatmap(corr, annot = True)
        plt.show()

        sns.pairplot(self.df, hue = "purchased")
        plt.show()

        cols = ["age", "estimated_salary", "purchased"]
        for i in range(len(cols)):
            plt.subplot(1, 3, i + 1)
            sns.boxplot(y = self.df[cols[i]])
            plt.title(cols[i])
        plt.show()

        for i in range(len(cols) - 1):
            plt.subplot(1, 2, i + 1)
            sns.histplot(self.df[cols[i]], kde = True)
            plt.title(cols[i])
        plt.show()

        print("EDA performed Successfully.", end = seperator)

    def split_data(self):
        """
        Split data
        Returns - None
        -------
        """

        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = self.random_state)

        print("Data Splitting Successful.", end = seperator)

    def transform_data(self):
        """
        Scale data
        Returns - None
        -------
        """

        self.X_train["gender"] = self.encoder.fit_transform(self.X_train["gender"])
        self.X_test["gender"] = self.encoder.transform(self.X_test["gender"])

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("Data Transformation Successful.", end = seperator)

    def train_model(self):
        """
        Train model
        Returns - None
        -------
        """

        self.model.fit(self.X_train, self.y_train)

        print("Model Trained Successfully.", end = seperator)

    def evaluate_model(self):
        """
        Evaluate model
        Returns - None
        -------
        """

        self.y_pred = self.model.predict(self.X_test)
        print(f"Accuracy Score = {accuracy_score(self.y_test, self.y_pred) * 100 : .2f} %")
        print(f"\nClassification Report = \n{classification_report(self.y_test, self.y_pred)}\n")
        print(f"Confusion Matrix = \n{confusion_matrix(self.y_test, self.y_pred)}\n")

        print("Model Evaluation Successful.", end=seperator)

def main():
    """
    Main function to run SVM Classifier model
    Returns - None
    -------
    """

    svc = SupportVectorClassifier("user_data.csv")
    svc.load_data()
    svc.check_data()
    svc.preprocess_data()
    svc.eda()
    svc.split_data()
    svc.transform_data()
    svc.train_model()
    svc.evaluate_model()

if __name__ == "__main__":
    main()