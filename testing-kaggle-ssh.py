from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os, pandas as pd

api = KaggleApi()
api.authenticate()
to_download = ["customers.csv", "transactions_train.csv", "articles.csv"]
pd_objects = []

for i, file in enumerate(to_download):
    print("Downlodaing file", file)
    api.competition_download_file(competition="h-and-m-personalized-fashion-recommendations",file_name=file)

    zf = ZipFile(f"{file}.zip")
    zf.extractall()
    zf.close()

    # Customers-file does not require dtype specification, the others do
    if i==0:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file, dtype={"article_id": str})
    pd_objects.append(df)
    # Remove files created from download now that it's loaded to memory
    os.remove(file)
    os.remove(f"{file}.zip")
    print("Sucessfully loaded and removed", file)