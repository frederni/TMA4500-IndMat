# TMA4500-IndMat
*Specialization project of Industrial Mathematics*

The explorative project focused on recommender systems for retail data, in particular [this dataset from H&M](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview).


**Files of interest**

* `explore_data.ipynb`: Exploratory Data Analysis (EDA) of the dataset
* `predict_collab_filtering.ipynb`: Started process of memory-based CF but never reached any conclusion
* `collaborative-filtering.ipynb`: Model-based collaborative filtering:
    1) Recommend the *k* most purchased items for all customers
    2) Recommend the *k* most purchased items within each user's most popular `index_group_no`
    3) Baseline machine learning model with transactional data
    4) Extension to model above, connecting side information with multilayer perceptron
* `utils/metrics.py`: Implementation of the MAP@12 metric
