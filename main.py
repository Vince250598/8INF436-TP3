import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from AutoEncoders import linearDoubleLayer, linearTripleLayer, nonLinearTripleLayer, linear2LayersWithSparsity

from DataframeAnalyser import AnalyseDataframe
from DataframePreprocesser import preprocessDataframe, train_test_validation_split
from Tools import computeAnomalyScores, computeResults

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

random_state = 2021

data = pd.read_csv('healthcare-dataset-stroke-data.csv')

AnalyseDataframe(data)

X, y = preprocessDataframe(data)

print(X.info())

# Séparation des données en groupe d'entrainement, de validation et de test

# Séparation 60/20/20
X_train, X_test, X_val, y_train, y_test, y_val = train_test_validation_split(X=X, y=y, test_percentage=0.2, validation_percentage=0.2, random_state=random_state)

# Séparation 70/15/15
# X_train, X_test, X_val, y_train, y_test, y_val = train_test_validation_split(X=X, y=y, test_percentage=0.15, validation_percentage=0.15, random_state=random_state)


'''
print("\nAuto-encodeur linéaire complet à 2 couches")
linearDoubleLayer(X_train, X_test, y_test, X_val, y_val, len(X_train.columns), printTestScores=True)


print("\nAuto-encodeurs linéaires incomplets à 2 couches")
scores = {}
for layer1Units in range(2, 9):
    print("La couche 1 possède " + str(layer1Units) + " unités")
    scores[layer1Units] = linearDoubleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires incomplets à 2 couches:")
print(scores)

print("\nAuto-encodeurs linéaires trop complets à 2 couches")
scores = {}
for layer1Units in range(11, 25):
    print("La couche 1 possède " + str(layer1Units) + " unités")
    scores[layer1Units] = linearDoubleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires trop complets à 2 couches:")
print(scores)
'''


'''
print("\nAuto-encodeurs linéaires incomplets à 3 couches")
scores = {}
for layer1Units in range(2, 9):
    for layer2Units in range(2, 9):
        print("La couche 1 possède " + str(layer1Units) + " unités")
        print("La couche 2 possède " + str(layer2Units) + " unités")
        key = str(layer1Units) + "," + str(layer2Units)
        scores[key] = linearTripleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units, layer2Units)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires incomplets à 3 couches:")
print(scores)
'''


'''
print("\nAuto-encodeurs non-linéaires incomplets à 3 couches")
scores = {}
for layer1Units in range(2, 9):
    for layer2Units in range(2, 9):
        print("La couche 1 possède " + str(layer1Units) + " unités")
        print("La couche 2 possède " + str(layer2Units) + " unités")
        key = str(layer1Units) + "," + str(layer2Units)
        scores[key] = nonLinearTripleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units, layer2Units)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs non-linéaires incomplets à 3 couches:")
print(scores)
'''


'''
print("\nAuto-encodeurs linéaires incomplets à 2 couches avec sparsity")
scores = {}
for layer1Units in range(2, 9):
    print("La couche 1 possède " + str(layer1Units) + " unités")
    scores[layer1Units] = linear2LayersWithSparsity(X_train, X_test, y_test, X_val, y_val, layer1Units)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires incomplets à 2 couches avec sparsity:")
print(scores)
'''


'''
print("\nAuto-encodeurs linéaires trop complets à 2 couches avec sparsity")
scores = {}
for layer1Units in range(11, 25):
    print("La couche 1 possède " + str(layer1Units) + " unités")
    scores[layer1Units] = linear2LayersWithSparsity(X_train, X_test, y_test, X_val, y_val, layer1Units)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires trop complets à 2 couches avec sparsity:")
print(scores)
'''


'''
print("\nAuto-encodeurs linéaires trop complets à 2 couches avec sparsity et dropout de 0.05")
scores = {}
for layer1Units in range(11, 25):
    print("La couche 1 possède " + str(layer1Units) + " unités")
    scores[layer1Units] = linear2LayersWithSparsity(X_train, X_test, y_test, X_val, y_val, layer1Units, dropoutPercentage=0.05)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires trop complets complets à 2 couches avec sparsity et dropout:")
print(scores)
'''


'''
print("\nAuto-encodeurs linéaires incomplets à 2 couches avec sparsity et dropout de 0.05")
scores = {}
for layer1Units in range(2, 9):
    print("La couche 1 possède " + str(layer1Units) + " unités")
    scores[layer1Units] = linear2LayersWithSparsity(X_train, X_test, y_test, X_val, y_val, layer1Units, dropoutPercentage=0.05)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires incomplets à 2 couches avec sparsity et dropout de 0.05:")
print(scores)
'''


'''
print("\nAuto-encodeurs linéaires incomplets à 2 couches avec dropout de 0.1")
scores = {}
for layer1Units in range(2, 9):
    print("La couche 1 possède " + str(layer1Units) + " unités")
    scores[layer1Units] = linearDoubleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units, dropoutPercentage=0.1)
print("Scores moyens(précision moyenne et écart-type) pour les auto-encodeurs linéaires incomplets à 2 couches avec dropout de 0.1:")
print(scores)
'''
