import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

seperator = f"\n\n{'--' * 70}\n\n"

class AprioriAlgorithm:
    """Implementation of Apriori Algorithm for Market Basket Analysis"""

    def __init__(self):
        self.df = None
        self.basket = None
        self.transactions = None
        self.encoder = TransactionEncoder()
        self.df_encoded = None
        self.algo = None
        self.rules = None

    def load_data(self):
        """
        Loads the raw dataset from a CSV file
        Returns - None
        -------
        """

        try:
            self.df = pd.read_csv("GroceriesDataset.csv")
            print(f"\nDataset = \n{self.df.head()}\n")

            print("Dataset loaded successfully", end = seperator)

        except FileNotFoundError:
            print("File not found")
        except Exception as error:
            print(error)

    def group_items(self):
        """
        Groups items by Member and Date to identify unique shopping transactions
        Returns - None
        -------
        """

        self.load_data()

        try:
            # Grouping by Member and Date to treat all items bought together as one 'basket'
            self.basket = self.df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()

            # Converting the 'itemDescription' column into a list of lists for the encoder
            self.transactions = self.basket["itemDescription"].tolist()
            print(f"Transactions Length = {len(self.transactions)}")

            print("\nSuccessfully grouped items by transactions", end = seperator)

        except Exception as error:
            print(error)

    def feature_encoding(self):
        """
        Converts the list of transactions into a One-Hot Encoded DataFrame
        Returns - None
        -------
        """

        self.group_items()

        try:
            # Performing One Hot Encoding to separate categorical transactions in boolean
            encoder_array = self.encoder.fit_transform(self.transactions)

            # Generating an encoded df
            self.df_encoded = pd.DataFrame(encoder_array, columns = self.encoder.columns_)

            print("Feature encoding successful", end = seperator)

        except Exception as error:
            print(error)

    def run_algorithm(self):
        """
        Identifies frequent item sets using the Apriori algorithm
        Returns - None
        -------
        """

        self.feature_encoding()

        try:
            # Running Apriori : min_support=0.01 means item must appear in at least 1% of transactions
            self.algo = apriori(
                self.df_encoded,
                min_support = 0.01,
                use_colnames = True
            )

            print(f"Total Frequent Item-sets = {self.algo.shape[0]}\n")

            print("Model Training Successful", end = seperator)

        except Exception as error:
            print(error)

    def generate_association_rules(self):
        """
        Generates rules (If X then Y) based on confidence and lift metrics
        Returns - None
        -------
        """

        self.run_algorithm()

        try:
            # Building rules from frequent item sets with a minimum confidence threshold of 10%
            rules_df = association_rules(
                self.algo,
                metric = "confidence",
                min_threshold = 0.1
            )

            if rules_df is None or rules_df.empty:
                print("No association rules found")
                return

            # Filtering rules to ensure both sides of the 'arrow' have at least one item
            """
                Antecedents : The "If" part (the item already in the basket)
                Consequents : The "Then" part (the item they are likely to add)
                This line ensures that both sides of the "If-Then" statement actually contain at least one item.
                It prevents "ghost" rules from appearing in your results
            """
            self.rules = rules_df[
                rules_df['antecedents'].apply(lambda x : len(x) >= 1) &
                rules_df['consequents'].apply(lambda x : len(x) >= 1)
            ]

            print(f"Association Rules = {self.rules.shape[0]}\n")

            """
                Support : How often the combination (A + B) appears in the entire dataset
                Lift > 1 : There is a strong positive relationship (A actually causes people to buy B)
                Lift = 1: The relationship is just random chance
            """
            print(f"Association Rules = \n{self.rules[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ].head()}\n")

            print("Generated association rules successfully", end = seperator)

        except Exception as error:
            print(error)

    def visualize_model(self):
        """
        Creates a visual summary of the most popular items in the dataset
        Returns - None
        -------
        """

        self.generate_association_rules()

        try:
            # Counting occurrences of each item to identify 'bestsellers'
            top_items = self.df["itemDescription"].value_counts().head(10)

            # Plotting the bar graph with top most purchased items
            top_items.plot(kind = "bar", title = "Top 10 Most Purchased Items")
            plt.xlabel("Item")
            plt.ylabel("Count")
            plt.show()

            print("Visualized model successfully", end = seperator)

        except Exception as error:
            print(error)

def main():
    """
    Initializes the apriori algorithm and runs it
    Returns - None
    -------
    """

    algorithm = AprioriAlgorithm()
    algorithm.visualize_model()

if __name__ == "__main__":
    main()