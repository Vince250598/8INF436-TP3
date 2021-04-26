import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def AnalyseDataframe(data):
    printBasicInformation(data)
    printBoxGraphs(data)


def printBasicInformation(data):
    # Impression des informations sur le dataset
    print(data.head())
    print(data.info())
    print(data.describe())

    # Compte du nombre de valeurs pour chacune des classes
    print(data['stroke'].value_counts())

    # Compte du nombre de valeurs manquantes
    print("Missing values per attribute: \n", data.isnull().sum())


def printBoxGraphs(data):
    for column in data:
        if data[column].dtype == np.float64:
            plt.subplots(figsize=(7, 5))
            sns.set_style('white')
            sns.despine()
            sns.boxplot(x=column, data=data)
            plt.title("Dispersion de l'attribut " + column)
            plt.tight_layout()
            plt.show()
