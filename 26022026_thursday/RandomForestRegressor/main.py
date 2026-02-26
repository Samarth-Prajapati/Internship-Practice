import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

seperator = f"\n\n{'--' * 75}\n\n"

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    filename = 'log_regressor.log',
    filemode = 'w'
)

class Regressor:
    """
    PERFORMING RANDOM FOREST REGRESSOR MODEL ON FLIGHT PRICE PREDICTION DATASET
    """

    def __init__(self, path):
        self.path = path
        self.df = None
        self.logger = logging.getLogger("RANDOM_FOREST_REGRESSOR")
        self.preprocessing = None
        self.ohe = OneHotEncoder(
            drop = "first",
            handle_unknown = "ignore"
        )
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.random_state = 42
        self.test_size = 0.3
        self.pipeline = None
        self.model = RandomForestRegressor(
            n_estimators = 23,
            max_depth = 10,
            verbose = 2,
            min_samples_split = 5,
            random_state = self.random_state
        )
        self.y_pred = None

    def load_dataset(self):
        """
        LOADING DATASET
        Returns - None
        -------
        """

        try:
            self.logger.info("Start Loading Dataset")

            self.df = pd.read_csv(self.path)
            print(f"\nDataset = \n\n{self.df.head()}\n")

            print("------> Dataset Loaded Successfully", end = seperator)
            self.logger.info("Dataset Loaded Successfully")

        except FileNotFoundError:
            print("File Not Found")
            self.logger.error("File Not Found")

        except Exception as error:
            print(error)
            self.logger.error(error)

    def check_dataset(self):
        """
        CHECKING DATASET
        Returns - None
        -------
        """

        self.logger.info("Start Analyzing Dataset")

        print("Dataset Info = \n")
        self.df.info()

        print(f"\nDataset Description = \n\n{self.df.describe()}\n")
        print(f"Shape of Dataset = {self.df.shape}\n")
        print(f"Sum of Duplicates = {self.df.duplicated().sum()}\n")
        print(f"Null Values in Dataset = \n\n{self.df.isnull().sum()}\n")

        print("------> Dataset Analysis Successfully", end = seperator)
        self.logger.info("Dataset Analysis Successfully")

    def preprocess(self):
        """
        PREPROCESSING DATASET
        Returns - None
        -------
        """

        self.logger.info("Start Preprocessing Dataset")

        self.df.drop(self.df.columns[0], axis = 1, inplace = True)
        self.df.drop("flight", axis = 1, inplace = True)
        print("Removed Unwanted Columns\n")
        print(f"Dataset = \n\n{self.df.head()}\n")

        print("------> Dataset Preprocessed Successfully", end = seperator)
        self.logger.info("Dataset Preprocessed Successfully")

    def eda(self):
        """
        PERFORMING EDA ON DATASET
        Returns - None
        -------
        """

        self.logger.info("Start Performing EDA on Dataset")

        corr = self.df.corr(numeric_only = True)
        sns.heatmap(corr, annot = True)
        plt.show()

        cols = ["duration", "days_left", "price"]
        for i in range(len(cols)):
            plt.subplot(1, 3, i + 1)
            sns.boxplot(y = self.df[cols[i]])
            plt.title(cols[i])
        plt.show()

        for i in range(len(cols)):
            plt.subplot(1, 3, i + 1)
            sns.histplot(self.df[cols[i]], kde = True)
            plt.title(cols[i])
        plt.show()

        print("------> EDA Performed Successfully", end = seperator)
        self.logger.info("EDA Performed Successfully")

    def handle_outliers(self):
        """
        HANDLE OUTLIERS
        Returns - None
        -------
        """

        self.logger.info("Start Handling Outliers")

        print("Outliers Info = \n")

        q1 = self.df["duration"].quantile(0.25)
        q3 = self.df["duration"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = self.df[(self.df["duration"] < lower) | (self.df["duration"] > upper)]
        print(f"Feature ( duration ) = {len(outliers)} outliers found.")

        self.df["duration"] = self.df["duration"].clip(lower = lower, upper = upper)

        plt.figure(figsize = (10, 12))
        sns.boxplot(y = self.df["duration"])
        plt.title("duration")
        plt.show()

        print("\n------> Outliers Handled Successfully", end = seperator)
        self.logger.info("Outliers Handled Successfully")

    def encoding(self):
        """
        ENCODING CATEGORICAL DATA
        Returns - None
        -------
        """

        self.logger.info("Start Encoding Categorical Data")

        categorical_cols = self.df.select_dtypes(include = ["str"]).columns
        self.preprocessing = ColumnTransformer(
            transformers = [("categorical", self.ohe, categorical_cols)],
            remainder="passthrough"
        )

        print("------> Encoding on Categorical Data Successful", end = seperator)
        self.logger.info("Encoding on Categorical Data Successful")

    def split_data(self):
        """
        SPLIT DATA
        Returns - None
        -------
        """

        self.logger.info("Start Splitting Data")

        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size = self.test_size,
            random_state=self.random_state
        )

        print("------> Data Splitting Successful", end = seperator)
        self.logger.info("Data Splitting Successful")

    def train_model(self):
        """
        TRAIN MODEL
        Returns - None
        -------
        """

        self.logger.info("Start Training Model")

        self.pipeline = Pipeline(
            steps = [
                ("preprocessor", self.preprocessing),
                ("model", self.model)
            ]
        )
        self.pipeline.fit(self.X_train, self.y_train)

        print("\n------> Model Training Successful", end=seperator)
        self.logger.info("Model Training Successful")

    def evaluate_model(self):
        """
        EVALUATE MODEL
        Returns - None
        -------
        """

        self.logger.info("Start Evaluating Model")

        self.y_pred = self.pipeline.predict(self.X_test)

        print(f"\nR2 Score = {r2_score(self.y_test, self.y_pred)}")
        print(f"Mean Absolute Error = {mean_absolute_error(self.y_test, self.y_pred)}")
        print(f"Mean Squared Error = {mean_squared_error(self.y_test, self.y_pred)}")
        print(f"Root Mean Squared Error = {root_mean_squared_error(self.y_test, self.y_pred)}")

        print("\n------> Model Evaluation Successful", end=seperator)
        self.logger.info("Model Evaluation Successful")

def main():
    """
    ALL OPERATIONS OF REGRESSOR CLASS ARE COMPLETED HERE
    Returns - None
    -------
    """

    path = "flight_price_prediction.csv"
    regressor = Regressor(path)

    regressor.load_dataset()
    regressor.check_dataset()
    regressor.preprocess()
    regressor.eda()
    regressor.handle_outliers()
    regressor.encoding()
    regressor.split_data()
    regressor.train_model()
    regressor.evaluate_model()

if __name__ == "__main__":
    main()