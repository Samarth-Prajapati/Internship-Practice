import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Classification:

    def __init__(self, path):
        self.df = None
        self.path = path
        self.le = LabelEncoder()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.random_state = 42
        self.test_size = 0.3
        self.model = RandomForestClassifier(
            max_depth = 2,
            min_samples_split = 5,
            min_samples_leaf = 2,
            ccp_alpha = 0.01,
            random_state = self.random_state)
        self.y_pred = None

    def load_data(self):
        """
        Loading dataset.
        Returns - None
        -------
        """

        self.df = pd.read_csv(self.path)
        print(f"DataFrame = \n\n{self.df.head()}")

        print("\n-----> Dataset Loaded Successfully.")

    def check_dataset(self):
        """
        Checking dataset.
        Returns - None
        -------
        """

        print("DataFrame Info = \n")
        self.df.info()

        print(f"\nDataFrame Description = \n\n{self.df.describe()}\n")

        print(f"DataFrame Shape = {self.df.shape}\n")

        print(f"Number of Duplicates in DataFrame = {self.df.duplicated().sum()}")

        print("\n-----> Dataset Checked Successfully.")

    def eda(self):
        """
        Performing EDA
        Returns - None
        -------
        """

        corr = self.df.corr(numeric_only = True)
        sns.heatmap(corr, annot = True)
        plt.show()

        cols = self.df.columns.tolist()
        cols = cols[:len(cols) - 1]
        for i in range(len(cols)):
            plt.subplot(3, 3, i + 1)
            sns.boxplot(y = self.df[cols[i]])
        plt.show()

        for i in range(len(cols)):
            plt.subplot(3, 3, i + 1)
            sns.histplot(self.df[cols[i]], kde = True)
        plt.show()

        print("-----> EDA Performed Successfully.")

    def handle_outliers(self):
        """
        Handle outliers
        Returns
        -------
        """

        print("Outliers Info = \n")

        cols = self.df.columns.tolist()
        cols = cols[:len(cols) - 1]
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            print(f"Feature ( {col} ) = {len(outliers)} outliers found.")

            self.df[col] = self.df[col].clip(lower = lower, upper = upper)

        for i in range(len(cols)):
            plt.subplot(3, 3, i + 1)
            sns.boxplot(y = self.df[cols[i]])
            plt.title(cols[i])
        plt.show()

        print("\n-----> Outliers Handled Successfully.")

    def encoding(self):
        """
        Encoding the data.
        Returns - None
        -------
        """

        self.df["label"] = self.le.fit_transform(self.df["label"])

        print("-----> Encoded Successfully.")

    def split_data(self):
        """
        Splitting the data.
        Returns - None
        -------
        """

        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size = self.test_size,
            random_state = self.random_state
        )

        print("-----> Split Data Successfully.")

    def train_model(self):
        """
        Training the model.
        Returns
        -------
        """

        self.model.fit(self.X_train, self.y_train)

        print("-----> Trained Model Successfully.")

    def check_model(self):
        """
        Checking model.
        Returns
        -------
        """

        self.y_pred = self.model.predict(self.X_test)

        print(f"ACCURACY SCORE = {accuracy_score(self.y_test, self.y_pred) * 100 : .2f} %\n")
        print(f"CLASSIFICATION REPORT = \n\n{classification_report(self.y_test, self.y_pred)}\n")
        print(f"CONFUSION MATRIX = \n\n{confusion_matrix(self.y_test, self.y_pred)}")

        print("\n-----> Model Checked Successfully.")

def main():
    print("========================================== RANDOM FOREST CLASSIFIER ===========================================")
    random_forest = Classification("CropRecommendation.csv")

    print("\n================================================= LOAD DATASET ================================================\n")
    random_forest.load_data()

    print("\n================================================ CHECK DATASET ================================================\n")
    random_forest.check_dataset()

    print("\n===================================================== EDA =====================================================\n")
    random_forest.eda()

    print("\n=============================================== HANDLE OUTLIERS ===============================================\n")
    random_forest.handle_outliers()

    print("\n=========================================== ENCODING TARGET FEATURE ===========================================\n")
    random_forest.encoding()

    print("\n================================================ SPLITTING DATA ===============================================\n")
    random_forest.split_data()

    print("\n================================================ TRAINING MODEL ===============================================\n")
    random_forest.train_model()

    print("\n================================================== CHECK MODEL ================================================\n")
    random_forest.check_model()

    print("\n===============================================================================================================")

if __name__ == "__main__":
    main()