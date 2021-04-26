from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import numpy as np

from Tools import computeAnomalyScores, computeResults


# Pour complet layer1Units = nombre d'attributs du dataset, pour incomplet moins que dataset pour trop complet plus que dataset
def linearDoubleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units, printTestScores=False, verbose=0,
                      dropoutPercentage=0):
    test_scores = []
    for i in range(0, 10):
        model = Sequential()
        model.add(Dense(units=layer1Units, activation='linear', input_dim=len(X_train.columns)))
        if dropoutPercentage > 0:
            model.add(Dropout(dropoutPercentage))
        model.add(Dense(units=len(X_train.columns), activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(x=X_train, y=X_train, validation_data=(X_val, y_val), epochs=10, batch_size=32,
                            shuffle=True, verbose=verbose)

        predictions = model.predict(X_test, verbose=1)
        anomalyScores = computeAnomalyScores(X_test, predictions)
        preds, avgPrecision = computeResults(trueLabels=y_test, anomalyScores=anomalyScores, returnPreds=True,
                                             printPlots=False)
        test_scores.append(avgPrecision)

    if printTestScores:
        print("Mean average precision over 10 runs: ", np.mean(test_scores))
        print("Coefficient of variation over 10 runs:",
              {round(np.std(test_scores) / np.mean(test_scores), 4)})
        print(test_scores)
    return np.mean(test_scores), round(np.std(test_scores) / np.mean(test_scores), 4)


def linearTripleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units, layer2Units, printTestScores=False,
                      verbose=0):
    test_scores = []
    for i in range(0, 10):
        model = Sequential()
        model.add(Dense(units=layer1Units, activation='linear', input_dim=len(X_train.columns)))
        model.add(Dense(units=layer2Units, activation='linear'))
        model.add(Dense(units=len(X_train.columns), activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(x=X_train, y=X_train, validation_data=(X_val, y_val), epochs=10, batch_size=32,
                            shuffle=True, verbose=verbose)

        predictions = model.predict(X_test, verbose=1)
        anomalyScores = computeAnomalyScores(X_test, predictions)
        preds, avgPrecision = computeResults(trueLabels=y_test, anomalyScores=anomalyScores, returnPreds=True,
                                             printPlots=False)
        test_scores.append(avgPrecision)

    if printTestScores:
        print("Mean average precision over 10 runs: ", np.mean(test_scores))
        print("Coefficient of variation over 10 runs:",
              {round(np.std(test_scores) / np.mean(test_scores), 4)})
        print(test_scores)
    return np.mean(test_scores), round(np.std(test_scores) / np.mean(test_scores), 4)


def nonLinearTripleLayer(X_train, X_test, y_test, X_val, y_val, layer1Units, layer2Units, printTestScores=False,
                         verbose=0):
    test_scores = []
    for i in range(0, 10):
        model = Sequential()
        model.add(Dense(units=layer1Units, activation='relu', input_dim=len(X_train.columns)))
        model.add(Dense(units=layer2Units, activation='relu'))
        model.add(Dense(units=len(X_train.columns), activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(x=X_train, y=X_train, validation_data=(X_val, y_val), epochs=10, batch_size=32,
                            shuffle=True, verbose=verbose)

        predictions = model.predict(X_test, verbose=1)
        anomalyScores = computeAnomalyScores(X_test, predictions)
        preds, avgPrecision = computeResults(trueLabels=y_test, anomalyScores=anomalyScores, returnPreds=True,
                                             printPlots=False)
        test_scores.append(avgPrecision)

    if printTestScores:
        print("Mean average precision over 10 runs: ", np.mean(test_scores))
        print("Coefficient of variation over 10 runs:",
              {round(np.std(test_scores) / np.mean(test_scores), 4)})
        print(test_scores)
    return np.mean(test_scores), round(np.std(test_scores) / np.mean(test_scores), 4)


def linear2LayersWithSparsity(X_train, X_test, y_test, X_val, y_val, layer1Units, printTestScores=False, verbose=0,
                              dropoutPercentage=0):
    test_scores = []
    for i in range(0, 10):
        model = Sequential()
        model.add(Dense(units=layer1Units, activation='linear', activity_regularizer=regularizers.l1(10e-5),
                        input_dim=len(X_train.columns)))
        if dropoutPercentage > 0:
            model.add(Dropout(dropoutPercentage))
        model.add(Dense(units=len(X_train.columns), activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(x=X_train, y=X_train, validation_data=(X_val, y_val), epochs=10, batch_size=32,
                            shuffle=True, verbose=verbose)

        predictions = model.predict(X_test, verbose=1)
        anomalyScores = computeAnomalyScores(X_test, predictions)
        preds, avgPrecision = computeResults(trueLabels=y_test, anomalyScores=anomalyScores, returnPreds=True,
                                             printPlots=False)
        test_scores.append(avgPrecision)

    if printTestScores:
        print("Mean average precision over 10 runs: ", np.mean(test_scores))
        print("Coefficient of variation over 10 runs:",
              {round(np.std(test_scores) / np.mean(test_scores), 4)})
        print(test_scores)
    return np.mean(test_scores), round(np.std(test_scores) / np.mean(test_scores), 4)
