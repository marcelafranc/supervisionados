import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer

breastCancer = load_breast_cancer()

def printBreastCancer():
    df_breastCancer = pd.DataFrame(breastCancer.data, columns=breastCancer.feature_names)


    sns.scatterplot(x=df_breastCancer.iloc[:, 0], y=df_breastCancer.iloc[:, 1])
    plt.xlabel(breastCancer.feature_names[0])
    plt.ylabel(breastCancer.feature_names[1])
    plt.title("Visualização do Dataset BreastCancer sem Agrupamento")
    plt.show()

printBreastCancer()