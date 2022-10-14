from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import pandas as pd
import os, io, sys, json, random, datetime, torch

class HMData:
    def __init__(self, articles_file, customers_file, transactions_train, transactions_test) -> None:
        self.article_content = {}
        pass
    def read_articles(self, articles_file):
        df = pd.read_csv(articles_file)
        for row in df.iterrows():
            self.article_content[row['article_id']] = {
                "article_id": row['article_id'],
                
            }


""" Psudeocode for generating dataset to DataLoader


def genereate_set(total_cases, portion_negative):
	number_positive = total_cases*(1-portion_negative)
	number_negative = total_cases*portion_negative
	df_pos = pd.sample(all_info_cust, all_info_article, n=number_positive, source=transactions)
	num_written = 0
	while num_written < number_negative:
		selection = [random.choice(customers), random.choice(article)]
		if selection not in transactions:
			with open("tmp.csv", 'a') as f:
				f.write(all_info_cust, all_info_article, label=0\n)
			num_written += 1

And then, we just put every column as a Tensor or something, idk


"""