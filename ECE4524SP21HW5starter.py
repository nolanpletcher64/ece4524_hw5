# ECE4524 SP21 HW5
# Created by Creed Jones on April 4, 2021
import numpy as np
import pandas as pd
import sklearn.linear_model as linmod
import sklearn.preprocessing as preproc
import sklearn.model_selection as modelsel
import sklearn.metrics as metrics
import sklearn.neural_network as nnet
import sklearn.tree as tree
import warnings
import time
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
pd.options.mode.chained_assignment = None  # default='warn'

def dataFrameFromFile(filename):
    df = pd.read_excel(filename)        # read an Excel spreadsheet
    print('File ', filename, ' is of size ', df.shape)
    print(df.dtypes)
    # fix any datetime64 columns
    for label, content in df.items():
        if (pd.api.types.is_datetime64_ns_dtype(content.dtype)):
            df[label] = df[label].dt.strftime('%Y-%m-%d').tolist()
    return df

def create_unknowns(dframe, categorical_column):
    newcol = dframe[categorical_column]
    newcol.replace({np.nan:'Unknown'}, inplace=True)
    dframe[categorical_column] = newcol
    return newcol

def preprocDataFrame(df, IDLabels, TargetLabels, Categoricals, Binaries, Numericals, StratifyColumn):
    # start preparing the dataset - missing values, categoricals, etc.
    df['PlayType'].replace({'EXTRA POINT':'', 'TIMEOUT':'', 'TWO-POINT CONVERSION':'', 'NO PLAY':''})
    df.dropna(subset=(IDLabels + TargetLabels + ['Formation', 'PlayType']), inplace=True)
    newdf = df[Binaries + Numericals + StratifyColumn]

    # one-hot encoding of categoricals
    for label in Categoricals:
        catcolumns = pd.get_dummies(create_unknowns(df, label), prefix=label)
        newdf[catcolumns.columns] = catcolumns

    target = df[TargetLabels]
    return newdf, target

def printRegressionModelStats(model, testX, testy, names, targetName, outputPrefix):
    Ypred = model.predict(testX)
    (ntest, npred) = testX.shape
    r2 = metrics.r2_score(testy, Ypred)
    adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
    mse = metrics.mean_squared_error(testy, Ypred)
    print("\n\r%s: R2 = %f, adjR2 = %f, MSE = %f" % (outputPrefix, r2, adjr2, mse))
    if (hasattr(model, 'coef_')):
        coeff = model.coef_.ravel()
        print(targetName, " = %6.4f" % model.intercept_, end="", flush=True)
        for featCount in range(len(names)):
            print(" + %6.4f*%s" % (coeff[featCount], names[featCount]), end="", flush=True)

def main():
    # your code goes here
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

