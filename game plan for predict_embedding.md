# Train one epoch psudeo
* data is a collection of indices for transaction rows
e.g. `data[0]` is `(customerID0, articleID0, label0)`

* `customer id, article id = data[i][customer_id], data[i][article_id]`
* `customer row = df_customer[id == customer id]`
* `article row = df_article[id == article id]`
* `Find the index correstponding to this row for customerdf and articled`

* `pred = model(torch.Tensor(customer index, article index))`
* Continue the rest as it stands now.



## Changing Dataset object so this works:
We need to make a couple of changes to the Dataset objects...

* `HM_dataset(Dataset)`:
    * Change `generate_dataset()` so we get a df on form `(customerID, articleID, label)`
        * For both postive and negative case
        * Return a shuffled concat of these two dfs (df_postive and df_negative)
    * Dont bother merging the raw data

    * Change __len__ so it returns length of df from generate_dataset function
    * Change __getitem__ to also use this df
    * Consider using LabelEncoder on article IDs?