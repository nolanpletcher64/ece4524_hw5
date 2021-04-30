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
            
    return mse

def main():
    
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)    
    
    # Load dataframe from file and sample 20%
    df = dataFrameFromFile("pbp-2020.xlsx")
    dfSampled = df.sample(None, 0.2, False, None, 42, None)
    
    # Add new column of seconds left
    dfSampled["SecondsLeft"] = (15 * (4 - dfSampled["Quarter"]) + dfSampled["Minute"]) * 60 + dfSampled["Second"]
    
    # Remove unnecessary columns
    dfSampled = dfSampled.drop('Quarter', 'columns')
    dfSampled = dfSampled.drop('Minute', 'columns')
    dfSampled = dfSampled.drop('Second', 'columns')
    
    # Process the dataframe
    dfProc, target = preprocDataFrame(dfSampled, ["GameId"], ["Yards"], ["Formation", "PlayType", "PassType", "RushDirection"], ["IsRush", "IsPass"], ["SecondsLeft", "Down", "ToGo", "YardLine"], [])
    
    # Normalize the columns to (-1, 1)
    scalar = preproc.MinMaxScaler((-1,1))
    dfProc = scalar.fit_transform(dfProc)
    
    # Split the dataframe into training and test partitions, split target too
    dfTrain, dfTest, targTrain, targTest = modelsel.train_test_split(dfProc, target, test_size=0.5, random_state=42)
    
    # Train linear regression model
    regm = linmod.Ridge()
    regm.fit(dfTrain, targTrain)
    
    # Test regression on training dataframe
    printRegressionModelStats(regm, dfTrain, targTrain, ["Formation", "PlayType", "PassType", "RushDirection", "IsRush", "IsPass", "SecondsLeft", "Down", "ToGo", "YardLine"], "Yards", "linregTrain")
    
    # Test regression on testing dataframe
    printRegressionModelStats(regm, dfTest, targTest, ["Formation", "PlayType", "PassType", "RushDirection", "IsRush", "IsPass", "SecondsLeft", "Down", "ToGo", "YardLine"], "Yards", "linregTest")
    
    # Train decision tree regression model
    dtReg = tree.DecisionTreeRegressor(max_depth=2)
    dtReg.fit(dfTrain, targTrain)
    
    # Test decision tree regression on training dataframe
    printRegressionModelStats(dtReg, dfTrain, targTrain, ["Formation", "PlayType", "PassType", "RushDirection", "IsRush", "IsPass", "SecondsLeft", "Down", "ToGo", "YardLine"], "Yards", "decregTrain")
    
    # Test decision tree regression on testing dataframe
    printRegressionModelStats(dtReg, dfTest, targTest, ["Formation", "PlayType", "PassType", "RushDirection", "IsRush", "IsPass", "SecondsLeft", "Down", "ToGo", "YardLine"], "Yards", "decregTest")    
        
    
    # For various numbers of hidden nodes in the first hidden layer
    for numFirst in [0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100]:
        # For various numbers of hidden nodes in the second hidden layer
        for numSecond in [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100]:
            
            
            # Construct and fit MLP 1
            if (numFirst == 0):
                regMLP1 = nnet.MLPRegressor(hidden_layer_sizes=(numSecond))
            else:
                regMLP1 = nnet.MLPRegressor(hidden_layer_sizes=(numFirst, numSecond))
            regMLP1.fit(dfTrain, targTrain)
            
            # Calculate MSE of MLP 1 on train data
            Ypred = regMLP1.predict(dfTrain)
            (ntest, npred) = dfTrain.shape
            r2 = metrics.r2_score(targTrain, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE1train = metrics.mean_squared_error(targTrain, Ypred)
            
            # Calculate MSE of MLP 1 on test data
            Ypred = regMLP1.predict(dfTest)
            (ntest, npred) = dfTest.shape
            r2 = metrics.r2_score(targTest, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE1test = metrics.mean_squared_error(targTest, Ypred)
            
            
            # Construct and fit MLP 2
            if (numFirst == 0):
                regMLP2 = nnet.MLPRegressor(hidden_layer_sizes=(numSecond))
            else:
                regMLP2 = nnet.MLPRegressor(hidden_layer_sizes=(numFirst, numSecond))
            regMLP2.fit(dfTrain, targTrain)
            
            # Calculate MSE of MLP 2 on train data
            Ypred = regMLP2.predict(dfTrain)
            (ntest, npred) = dfTrain.shape
            r2 = metrics.r2_score(targTrain, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE2train = metrics.mean_squared_error(targTrain, Ypred)
            
            # Calculate MSE of MLP 2 on test data
            Ypred = regMLP2.predict(dfTest)
            (ntest, npred) = dfTest.shape
            r2 = metrics.r2_score(targTest, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE2test = metrics.mean_squared_error(targTest, Ypred)   
            
            
            # Construct and fit MLP 3
            if (numFirst == 0):
                regMLP3 = nnet.MLPRegressor(hidden_layer_sizes=(numSecond))
            else:
                regMLP3 = nnet.MLPRegressor(hidden_layer_sizes=(numFirst, numSecond))            
            regMLP3.fit(dfTrain, targTrain)
            
            # Calculate MSE of MLP 3 on train data
            Ypred = regMLP3.predict(dfTrain)
            (ntest, npred) = dfTrain.shape
            r2 = metrics.r2_score(targTrain, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE3train = metrics.mean_squared_error(targTrain, Ypred)                
            
            # Calculate MSE of MLP 3 on test data
            Ypred = regMLP3.predict(dfTest)
            (ntest, npred) = dfTest.shape
            r2 = metrics.r2_score(targTest, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE3test = metrics.mean_squared_error(targTest, Ypred)            
            
            
            # Construct and fit MLP 4
            if (numFirst == 0):
                regMLP4 = nnet.MLPRegressor(hidden_layer_sizes=(numSecond))
            else:
                regMLP4 = nnet.MLPRegressor(hidden_layer_sizes=(numFirst, numSecond))            
            regMLP4.fit(dfTrain, targTrain)
            
            # Calculate MSE of MLP 4 on train data
            Ypred = regMLP4.predict(dfTrain)
            (ntest, npred) = dfTrain.shape
            r2 = metrics.r2_score(targTrain, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE4train = metrics.mean_squared_error(targTrain, Ypred)
            
            # Calculate MSE of MLP 4 on test data
            Ypred = regMLP4.predict(dfTest)
            (ntest, npred) = dfTest.shape
            r2 = metrics.r2_score(targTest, Ypred)
            adjr2 = 1 - ((ntest-1)/(ntest - npred - 1))*(1-r2)
            MSE4test = metrics.mean_squared_error(targTest, Ypred)            
            
            
            # Average MSE's and print
            avgMSEtrain = (MSE1train + MSE2train + MSE3train + MSE4train) / 4
            avgMSEtest = (MSE1test + MSE2test + MSE3test + MSE4test) / 4            
            print("Average MSE for training data with (" + str(numFirst) + "," + str(numSecond) + ") nodes in the two hidden layers = " + str(avgMSEtrain))
            print("Average MSE for testing data with (" + str(numFirst) + "," + str(numSecond) + ") nodes in the two hidden layers = " + str(avgMSEtest))
            
    
    print("Finished running.")
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

