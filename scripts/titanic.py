###### Imports

### Pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import seaborn as sns
sns.set_style('whitegrid')
###%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def getDataBases(trainName, testName ):

    print "\n****************************"
    print "  Getting Data Bases"
    # get titanic & test csv files as a DataFrame
    train_df = pd.read_csv(trainName)
    test_df  = pd.read_csv(testName)

    return (train_df, test_df)


def printDF(dataF):

    print "*** Print DataFrame!!"
    print
    print "  First five rows:"
    print 
    print  dataF.head()
    print
    print "  Info:"
    dataF.info()
    print


def getTitle(name):

    if "mrs." in name.lower():
        return "Mrs."
    elif "mr." in name.lower():
        return "Mr."
    elif "miss." in name.lower():
        return "Miss."
    elif "master" in name.lower():
        return "Master."
    else:
        return "Hmm..."

def addTicketLetter(df):

    df['ticketLetter'] = df.apply( lambda row: row['Ticket'][0], axis=1)
    
def addTitleApply(df):

    df['Title'] = df.apply (lambda row: getTitle(row["Name"]) ,axis=1)
    
def addTitle(df):

    for i, row in df.iterrows():
        if i < 5: print row["Name"], getTitle(row["Name"])
        row["Title"]=getTitle(row["Name"])

    print df.head()


def doSex(dataFrame):

    dataFrame['SexIndex'] = dataFrame.apply( lambda row: (1 if (row['Sex']=="male") else 0), axis=1)
    
    
def doPlotSmall(dataFrame, category):

    if category not in list(dataFrame): return

    print "\n**********************"
    print " doPlotSmall"
    print " cat", category
    
    # only in dataFrame, fill the two missing values with the most occurred value, which is "S".
    #dataFrame[category] = dataFrame[category].fillna("S")

    # plot
    sns.factorplot(category,'Survived', data=dataFrame,size=4,aspect=3)

    fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

    # sns.factorplot(category,data=dataFrame,kind='count',order=['S','C','Q'],ax=axis1)
    # sns.factorplot('Survived',hue=category,data=dataFrame,kind='count',order=[1,0],ax=axis2)
    sns.countplot(x=category, data=dataFrame, ax=axis1)
    sns.countplot(x='Survived', hue=category, data=dataFrame, order=[1,0], ax=axis2)

    # group by embarked, and get the mean for survived passengers for each value in Embarked
    embark_perc = dataFrame[[category, "Survived"]].groupby([category],as_index=False).mean()
    sns.barplot(x=category, y='Survived', data=embark_perc,ax=axis3)

    plotName="plots/output"+category+".png"
    fig.savefig(plotName)
    print " output to", plotName

    
def doPlotLarge(dataFrame, category):

    if category not in list(dataFrame): return
    
    # Fare
    print " \n**********************"
    print " doPlotLarge"
    print " Category of ", category

    # only for test_df, since there is a missing category values
    try:
        dataFrame[category].fillna(dataFrame[category].median(), inplace=True)
    except:
        dataFrame[category].fillna(dataFrame[category].mode(), inplace=True)
        
    # convert from float to int
    dataFrame[category] = dataFrame[category].astype(int)

    facet = sns.FacetGrid(dataFrame, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,category,shade= True)
    facet.set(xlim=(0, dataFrame[category].max()))
    facet.add_legend()
    
    plotName="plots/output"+category+".png"
    facet.savefig(plotName)
    print " output to ", plotName


def prepTrainSet(dataFrame,toTrainFor,cuts=False):

    print "\n*****************"
    print "Prep training set"
    X_train = dataFrame.drop(toTrainFor, axis=1)
    if cuts:
        for cut in cuts:
            if cut not in list(dataFrame): continue
            print "   Drop",cut
            X_train = X_train.drop(cut,axis=1)

    Y_train=dataFrame[toTrainFor]

    return (X_train, Y_train)

def doML(X_train_raw, Y_train_raw, type):

    if type=="logreq":
        print "Doing logistic regression"
        machineLearner = LogisticRegression()
        
    elif type=="randomForest":
        print "Doing random forest"
        machineLearner = RandomForestClassifier(n_estimators=100)
        
    else:
        print "No machine learner option"
        return

    X_train, X_test, Y_train, Y_test = train_test_split(X_train_raw, Y_train_raw, test_size=0.4,random_state=3)
    
    machineLearner.fit(X_train, Y_train)
    machineLearner.score(X_train, Y_train)
    print machineLearner.score(X_test, Y_test)
    

        
def main():

    trainName="data/train.csv"
    testName="data/test.csv"
    #trainName=testName

    train_df, test_df = getDataBases(trainName, testName)
    printDF(train_df)
    printDF(test_df)
    ##addTitleApply(train_df)    
    ##addTicketLetter(train_df)\
    doSex(train_df)

    plotCols=["Title","Embarked","Sex","SibSp","Parch"]
    for col in plotCols:
        print col
        doPlotSmall(train_df,col)

    plotLargeCols=["Fare", "Age"]
    for col in plotLargeCols:
        print col
        doPlotLarge(train_df, col)

        
    # define training and testing sets
    toCut=["Title","Embarked","PassengerId","Cabin","Ticket","Name","Sex"]
    X_train, Y_train = prepTrainSet(train_df,"Survived",toCut)

    print X_train.head()

    mlTypes=["randomForest","logreq"]
    print list(X_train)
    for mlType in mlTypes:
        doML(X_train, Y_train, mlType)

    for testCat in list(X_train):
        print "\n------------"
        print "Dropping", testCat,
        X_train_testCat = X_train.drop(testCat,axis=1)
        print list(X_train_testCat)
        for mlType in mlTypes:
            doML(X_train_testCat, Y_train, mlType)
    
        
if __name__ == "__main__":
    main()

    print " \n*******  Done *********\n"
