# Bad code jail
import pandas as pd
import numpy as np
from typing import Tuple

def sample_all_hm(
    num_users: int, k_positive: float = 0.10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sample a dataset with `num_user` customers without any unreferenced rows.

    Args:
        num_users (int): How many users to sample from
        TODO finish docstring params
        TODO make sure this works with complete dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrame of customers (1), transactions belonging to customers (2),
                                                            and articles belonging to transactions (3)
    """
    df_customers = naive_csv_sampler("dataset/customers.csv", num_users)
    df_transactions = pd.read_csv("dataset/transactions_train.csv")  # TODO slow!!

    df_transactions_positive = df_transactions[
        df_transactions["customer_id"].isin(df_customers["customer_id"])
    ]
    df_transactions_negative = (
        pd.merge(df_transactions, df_transactions_positive, indicator=True, how="outer")
        .query('_merge=="left_only"')
        .drop("_merge", axis=1)
    )
    df_transactions_negative = df_transactions_negative.sample(
        n=num_users / k_positive
    )  # Sample from negative labels

    df_transactions = pd.merge(
        df_transactions_positive, df_transactions_negative, indicator=True
    )
    df_articles = pd.read_csv("dataset/articles.csv")
    df_articles = df_articles[
        df_articles["article_id"].isin(df_transactions["article_id"])
    ]
    return df_customers, df_transactions, df_articles

if __name__ == '__main__':
    pass