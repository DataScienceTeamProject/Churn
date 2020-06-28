import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
warnings.filterwarnings(action='ignore')


df = pd.read_csv('Dataset_1.csv')

# Object Selection--> Discriminate Churn
# Discrimination based on all data and categorical data
###################################################################################33


#Inspection
# Function to output na value for each column
def foundNan(df):
    print(df.isna().any())
    print(df.isna().sum())

# Inspection
# Check if there are outliers through the plot.
# There were no outliers in this project.
##################################################

def plotSeries(data,key):
    plt.title(key)
    plt.hist(data[key])
    plt.show()

# Preprocessing and Inspection
# print K_best value of data.
# Data divided into categorical data can also be checked
#######################################
def select_k_best(df, n):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    # Num of Top feature select...
    bestFeatures = SelectKBest(score_func=chi2, k=n)
    fit = bestFeatures.fit(X, Y)
    dfColumns = pd.DataFrame(X.columns)
    dfscores = pd.DataFrame(fit.scores_)
    featureScores = pd.concat([dfColumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print(featureScores.nlargest(n, 'Score'))
    return featureScores.nlargest(n,'Score')



#Preprocessing
# Perform scaling according to the entered scaler.
def ScalingData(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return df_scaled

#Preprocessing
# Encoding String value to Numeric Value
def columnEncoding(col,df):
    for c in col:
        # get All distinguish Value of column
        c_unique = df[c].unique()
        print(c_unique)
        encoder = LabelEncoder()
        encoder.fit(df[c])
        df[c] = encoder.transform(df[c])
        result_unique = df[c].unique()
        print("%s is replaced %s"%(c_unique,result_unique)) # print replaced label compared origin label
    return df

#Preprocessing
# Part that determines how to handle na value of dataframe
def handleNa(col, df,work="mean"):
    # fill na value for 'mean'
    if work=="mean":
        for c in col:
            mean = df[c].mean()
            df[c]=df[c].fillna(mean)
    # fill na value for 'median'
    elif work=="median":
        for c in col:
            median = df[c].median()
            df[c]=df[c].fillna(median)
    # fill na value for 'mode'
    elif work=="mode":
        for c in col:
            mode = df[c].mode()[0]
            df[c]=df[c].fillna(mode)
    # drop row which contains na value
    elif work=="drop":
        df = df.dropna(subset=col)
    return df


print('--------------------------------------------------------------------------')
print(df.info())#info Get approximate information through output.
print(df)
# Preprocessing
# split_df split data by specific categorical value
# current deprecated
###################################################
def split_df(data,standard):
    std_list = data[standard].unique()
    data_list = []
    for s in std_list:
        data_list.append(data[data==s])
    return data_list

# deprecated
# current Not used
###################################################
def visualization(data):
    print(data)

split_df(df, 'Gender') # split data by gender of customer
foundNan(df)#nan exists, endcoding is impossible, and the value is replaced by the mode (categorical value)



df=df.drop(['Unnamed: 0','Surname'],axis=1)
# Since outlier can have a big influence on the statistical value of data, drop is performed.
# If you handle na the entire table by drop method, when the value of another column is na, it is dropped to another row.
#Because, drop is performed on one column each.
df = handleNa(['Exited'],df,"drop")
is_plot=False
if is_plot:
    for k in df:
        t = handleNa([k],df,"drop")
        plotSeries(t, k)


df=handleNa(['Geography','Gender'],df,work="mode")#mode로 categorical 제거...
foundNan(df)

for c in df:
    print("col %s : %d"%(c,len(df[c].unique())))
    print('----------------------------------------')
print('----------------------------------------')
df = columnEncoding(['Geography','Gender'],df)#Encoding을 통해 String vlaue를 제거

#-------------------------------------------------------------
#                Process & Actual Action
#-------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Evaluation
# calculate accuracy own way
def confirmAccuracy(gt, result):
    correct=0
    for i in range(len(gt)):
        if(gt[i]==result[i]):
            correct+=1
    return correct/len(gt)*100

# Evaluation
# evaluation predicted value using accuracy// confusion matrix// classification_report
def evaluation(gt, predict):
    print(confirmAccuracy(gt, predict))
    print(confusion_matrix(gt, predict))
    print(classification_report(gt,predict))

# Analysis
# predict and evaluation model.(with parameter) by evaluation dataset.
def predict(X_train, Y_train, X_ev, Y_ev, model, param, cv=10):
    print(param, 'params')
    gridSearchModel = GridSearchCV(model, param_grid=param, cv=cv, refit=True)
    gridSearchModel.fit(X_train,Y_train)
    print("Best Parameter: [%s]"%(gridSearchModel.best_params_))
    print("Best Score(test Set): [%f]"%(gridSearchModel.best_score_))
    fitted_model = gridSearchModel.best_estimator_
    predict = fitted_model.predict(X_train)
    print("Score with TrainSet: [%f]"%(confirmAccuracy(Y_train.values,predict)))
    evaluation(Y_train.values, predict)
    predict = fitted_model.predict(X_ev)
    print("Score with EvalSet: [%f]"%(confirmAccuracy(Y_ev.values, predict)))
    evaluation(Y_ev.values,predict)
    return fitted_model

#Analysis
# predict and evaluation model. (with parameter) by train set
def predict2(X, Y, model, param, cv=10):
    print(param, 'params')
    gridSearchModel = GridSearchCV(model, param_grid=param, cv=cv, refit=True)
    gridSearchModel.fit(X,Y)
    print("Best Parameter: [%s]"%(gridSearchModel.best_params_))
    print("Best Score(test Set): [%f]"%(gridSearchModel.best_score_))
    fitted_model = gridSearchModel.best_estimator_
    predict = fitted_model.predict(X)
    print("Score with TrainSet: [%f]"%(confirmAccuracy(Y.values,predict)))
    evaluation(Y.values, predict)
    return fitted_model

# Preprocess ~ Evaluation
# main part of Project. excute end to end process
# preprocessing ~ evaluation.
############################################################################
def end_to_end_process(scaler, na, algorithm,param , df,pca=None):
    df = handleNa(df.columns, df, work=na)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X = pd.DataFrame(ScalingData(X, scaler), columns=X.columns)
    if pca is not None:
        X = pca.fit_transform(X)
    '''
    # analysis of train, test set  
    X, X_eval, Y, Y_eval = train_test_split(X, Y, stratify=Y, random_state=8, test_size=.2)#using evaluation dataset.
    predict(X, Y, X_eval, Y_eval, algorithm, param)
    '''
    predict2(X,Y,algorithm,param)#analysis of tstSet

logistic = LogisticRegression()
knn = KNeighborsClassifier()
randomForest = RandomForestClassifier()
decisionTree=DecisionTreeClassifier(max_depth=10)#10
bagging = BaggingClassifier(decisionTree)
rf_params = {
    'n_estimators':np.arange(4, 30, 2),
    'criterion':['gini','entropy'],
    'max_depth':[None,1,2,3]
}
logisticParams ={
    'penalty':['l1','l2','elasticnet'],
    'dual':[True,False],
    'C':[0.25,0.5,1.0,2.0,3.0,4.0],
    'fit_intercept':[True,False]
}
knn_params={'n_neighbors':np.arange(1,5),
        'weights':['uniform','distance'],}

bg_params={'n_estimators':np.arange(4,20,2),
           'bootstrap':[True,False],}

r_scaler = RobustScaler()
s_scaler = StandardScaler()
m_scaler = MinMaxScaler()

df = df.drop(['HasCrCard'],axis=1)
#divide data by gender
df_male = df[df['Gender']==1]
df_female = df[df['Gender']==0]
#divide data by geography
df_france = df[df['Geography']==0]
df_spain = df[df['Geography']==2]
df_german = df[df['Geography']==1]
df_filled = handleNa(df.columns, df, work="mean")
df_filled=ScalingData(df_filled,s_scaler)
#df_x, df_y = df_filled.iloc[:,:-1], df_filled.iloc[:,-1]

# configuration part
# scaler_parameter:
# na_parameter:
# data: Data to perform train and test
# pca: None_pca not applied // int_pca's k value
####################################################################33
scaler_paramerter = s_scaler
na_parameter="mean"
data = df
# Variable to decide whether to use kbest or not
set_K_best = True

if set_K_best:
    # The number of variables can be adjusted by adjusting the second parameter
    index = select_k_best(df,10)['Specs'].to_numpy()
    index = index.tolist()
    index.append("Exited")
    data = data[index]

pca=None
# Variable determining whether to use PCA
pca_parameter = None
# When the PCA count is present, pca is initialized.
if pca_parameter is not None:
    pca = PCA(n_components=pca_parameter)

# Stat End to End process for...
# Logistic Regression // KNearestNeighbor // RandomForest // Bagging
###########################################################################################
print("------------------------------------------------------------")
print("------------------------------------------------------------")
# Logistic regression is a model that tries to predict the distribution of a model through a function similar to linear regression.
# However, there is a difference in predicting a function that divides classes,
# and Logistic Regression is considered to be more suitable for classification, and Logistic Regression is applied.
##############################################################################################
print("LogisticRegression")
end_to_end_process(scaler_paramerter, na_parameter, logistic, logisticParams, data,pca=pca)
print("------------------------------------------------------------")
# Knn is a very simple classification algorithm. I thought it was a suitable model for future ensemble learning and comparison.
##############################################################################################
print("K Nearest Neighbor")
end_to_end_process(scaler_paramerter, na_parameter, knn, knn_params, data,pca=pca)
print("------------------------------------------------------------")
# Random Forest:
# As a kind of ensemble learning technique
# This is a way to reduce variance by limiting variables in bagging.
##############################################################################################
print("RandomForest")
end_to_end_process(scaler_paramerter, na_parameter, randomForest, rf_params, data,pca=pca)
print("------------------------------------------------------------")
# Bagging:
# This is a technique to derive results by voting based on multiple Decision Trees.
# It has the advantage of reducing the variance of the model.
##############################################################################################
print('Bagging')
end_to_end_process(scaler_paramerter, na_parameter, bagging, bg_params, data, pca=pca)