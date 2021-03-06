import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import numpy as np


def computeAnomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - np.array(reducedDF)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=originalDF.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    return loss


def computeResults(trueLabels, anomalyScores, returnPreds=False, printPlots=True):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds = \
        precision_recall_curve(preds['trueLabel'], preds['anomalyScore'])
    average_precision = \
        average_precision_score(preds['trueLabel'], preds['anomalyScore'])
    if printPlots:
        plt.step(recall, precision, color='k', alpha=0.7, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: '
                  'Average Precision = {0:0.2f}'
                  .format(average_precision))
        fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['anomalyScore'])
        areaUnderROC = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='r', lw=2,
                 label='ROC curve')
        plt.plot([0, 1], [0, 1], color='k',
                 lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic: '
                  'Area under the curve = {0:0.2f}'
                  .format(areaUnderROC))
        plt.legend(loc="lower right")
        plt.show()
    if returnPreds == True:
        return preds, average_precision
    return average_precision
