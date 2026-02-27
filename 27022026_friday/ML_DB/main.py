import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

seperator = f"\n{'--' * 65}\n"

class DataBase:

    def __init__(self):
        try:
            self.engine = create_engine("mssql+pyodbc://localhost/ML?driver=ODBC+Driver+17+for+SQL+Server")
            print("\nDB Connected Successfully", end = seperator)

        except SQLAlchemyError:
            print("SQLAlchemy Error")
        except Exception as error:
            print(error)

    def load_table(self, table_name):
        """
        Load DB Table to dataframe
        Parameters
        ----------
        table_name - table name

        Returns - dataframe
        -------
        """

        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            print("Data converted to DataFrame", end = seperator)

            return df

        except Exception as error:
            print(error)

class Classification:

    def __init__(self, df):
        self.df = df
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

    def check_dataset(self):
        """
        Checking dataset.
        Returns - None
        -------
        """

        print(f"DataFrame = \n{self.df.head()}\n")
        print("DataFrame Info = ")
        self.df.info()

        print(f"\nDataFrame Description = \n{self.df.describe()}\n")
        print(f"DataFrame Shape = {self.df.shape}")
        print(f"Number of Duplicates in DataFrame = {self.df.duplicated().sum()}")

        print("\nDataset Checked Successfully", end = seperator)

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

        print("EDA Performed Successfully", end = seperator)

    def handle_outliers(self):
        """
        Handle outliers
        Returns
        -------
        """

        print("Outliers Info = ")

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

        print("\nOutliers Handled Successfully", end = seperator)

    def encoding(self):
        """
        Encoding the data.
        Returns - None
        -------
        """

        self.df["label"] = self.le.fit_transform(self.df["label"])

        print("Encoded Successfully", end = seperator)

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

        print("Split Data Successfully", end = seperator)

    def train_model(self):
        """
        Training the model.
        Returns
        -------
        """

        self.model.fit(self.X_train, self.y_train)

        print("Trained Model Successfully", end = seperator)

    def check_model(self):
        """
        Checking model.
        Returns
        -------
        """

        self.y_pred = self.model.predict(self.X_test)

        print(f"ACCURACY SCORE = {accuracy_score(self.y_test, self.y_pred) * 100 : .2f} %\n")
        print(f"CLASSIFICATION REPORT = \n{classification_report(self.y_test, self.y_pred)}\n")
        print(f"CONFUSION MATRIX = \n{confusion_matrix(self.y_test, self.y_pred)}")

        print("\nModel Checked Successfully", end = seperator)

def main():
    loader = DataBase()
    df = loader.load_table("dataset")

    classifier = Classification(df)
    classifier.check_dataset()
    classifier.eda()
    classifier.handle_outliers()
    classifier.encoding()
    classifier.split_data()
    classifier.train_model()
    classifier.check_model()

if __name__ == "__main__":
    main()
