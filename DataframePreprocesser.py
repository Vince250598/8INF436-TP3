from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


# La somme de train_percentage, test_percentage et validation_percentage doit être égale à 1
def train_test_validation_split(X, y, test_percentage, validation_percentage, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=random_state,
                                                        stratify=y)

    train_percentage = 1 - (validation_percentage + test_percentage)

    validation_size = validation_percentage / (validation_percentage + train_percentage)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                      random_state=random_state, stratify=y_train)

    return X_train, X_test, X_val, y_train, y_test, y_val


def preprocessDataframe(data):
    data = removeAberrantRows(data)

    # Suppression de la colonne id
    data = data.drop(['id'], axis=1)

    data = encodeCategoricalAttributes(data)

    # Séparation des données et des labels
    X = data.loc[:, data.columns != 'stroke']
    labels = data.stroke
    y = pd.Series(data=labels, name="stroke")

    X = scaleAttributes(X)

    return X, y


def removeAberrantRows(data):
    # Suppression des ligne ou la valeure de bmi est manquante
    data = data.dropna(subset=['bmi'])
    print("Missing values per attribute: \n", data.isnull().sum())

    # Suppression des ligne ou la valeure de bmi est superieur a 70
    data = data[data.bmi < 70]

    # Suppression des ligne ou la valeure de bmi est inferieur a 15
    data = data[data.bmi > 15]

    # Suppression de la ligne avec le genre Autre (Biais)
    data = data[data.gender != 'Other']

    return data


def encodeCategoricalAttributes(data):
    # Encodage des attributs catégoriques
    labelEncoder = LabelEncoder()
    data["gender"] = labelEncoder.fit_transform(data["gender"])
    data["ever_married"] = labelEncoder.fit_transform(data["ever_married"])
    data["work_type"] = labelEncoder.fit_transform(data["work_type"])
    data["Residence_type"] = labelEncoder.fit_transform(data["Residence_type"])
    data["smoking_status"] = labelEncoder.fit_transform(data["smoking_status"])

    return data


def scaleAttributes(X_train):
    # Mise a l'echelle des attributs
    scaler = StandardScaler()
    X_train.loc[:, :] = scaler.fit_transform(X_train)

    return X_train
