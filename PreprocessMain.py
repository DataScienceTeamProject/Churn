import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
warnings.filterwarnings(action='ignore')


df = pd.read_csv('Dataset_1.csv')
r_scaler = RobustScaler()
# Object Selection--> Discriminate Churn
# Discrimination based on all data and categorical data
###################################################################################33

# Perform scaling according to the entered scaler.
def ScalingData(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return df_scaled
# Function to output na value for each column
def foundNan(df):
    print(df.isna().sum())

def columnEncoding(col,df):
    for c in col:
        c_unique = df[c].unique()
        print(c_unique)
        encoder = LabelEncoder()
        encoder.fit(df[c])
        df[c] = encoder.transform(df[c])
        result_unique = df[c].unique()
        print("%s is replaced %s"%(c_unique,result_unique))
    return df
# Part that determines how to handle na value of dataframe
def handleNa(col, df,work="mean"):
    if work=="mean":
        for c in col:
            mean = df[c].mean()
            df[c]=df[c].fillna(mean)
    elif work=="median":
        for c in col:
            median = df[c].median()
            df[c]=df[c].fillna(median)
    elif work=="mode":
        for c in col:
            mode = df[c].mode()[0]
            df[c]=df[c].fillna(mode)
    elif work=="drop":
        df = df.dropna(subset=col)
    return df

print('--------------------------------------------------------------------------')
print(df.info())#info 출력을 통해 대략적 정보를 얻음.

# Check if there are outliers through the plot.
# There were no outliers in this project.
def plotSeries(data,key):
    plt.title(key)
    plt.hist(data[key])
    plt.show()

def split_df(data,standard):
    std_list = data[standard].unique()
    data_list = []
    for s in std_list:
        data_list.append(data[data==s])
    return data_list

split_df(df, 'Gender')
foundNan(df)#nan이 존재, Endcoding이 불가능한 값은 최빈값으로 대체 (카테고리컬 값은)



df=df.drop(['Unnamed: 0','Surname'],axis=1)
# outlier는 데이터의 통계 값에 큰 영향을 미칠 수 있기 때문에 drop을 수행한다.
# table 전체를 drop 방식으로 handle na 한다면, 다른 column의 값이 na일 때 다른 row까지 drop되고 만다.
# 때문에 drop은 하나의 Column에 각각 수행해 준다.
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
'''
df = handleNa(df.columns,df,work="mean") # 나머지 value들 mean으로 fillna
foundNan(df)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
X = pd.DataFrame(ScalingData(X,r_scaler),columns=X.columns)

from sklearn.model_selection import train_test_split
#
X, X_eval, Y, Y_eval = train_test_split(X,Y,stratify=Y,random_state=8, test_size=.2)
print(X.shape)
print(Y.shape)
print(X_eval.shape)
print(Y_eval.shape)
'''
#-------------------------------------------------------------
#                       Process
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


def confirmAccuracy(gt, result):
    correct=0
    for i in range(len(gt)):
        if(gt[i]==result[i]):
            correct+=1
    return correct/len(gt)*100

def evaluation(gt, predict):
    print(confirmAccuracy(gt, predict))
    print(confusion_matrix(gt, predict))
    print(classification_report(gt,predict))

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

def end_to_end_process(scaler, na, algorithm,param , df,pca=None):
    df = handleNa(df.columns, df, work=na)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X = pd.DataFrame(ScalingData(X, scaler), columns=X.columns)
    if pca is not None:
        X = pca.fit_transform(X)
    #X, X_eval, Y, Y_eval = train_test_split(X, Y, stratify=Y, random_state=8, test_size=.2)
    #predict(X, Y, X_eval, Y_eval, algorithm, param)
    predict2(X,Y,algorithm,param)

logistic = LogisticRegression()
knn = KNeighborsClassifier()
randomForest = RandomForestClassifier()
decisionTree=DecisionTreeClassifier(max_depth=10)
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

s_scaler = StandardScaler()
m_scaler = MinMaxScaler()
pca_2 = PCA(n_components=2)
pca_3 = PCA(n_components=3)
df = df.drop(['HasCrCard'],axis=1)
df_male = df[df['Gender']==1]
df_female = df[df['Gender']==0]
df_france = df[df['Geography']==0]
df_spain = df[df['Geography']==2]
df_german = df[df['Geography']==1]
df_filled = handleNa(df.columns, df, work="mean")
df_filled=ScalingData(df_filled,s_scaler)
#df_x, df_y = df_filled.iloc[:,:-1], df_filled.iloc[:,-1]

scaler_paramerter = s_scaler
na_parameter="drop"
pca=None
pca_parameter = 6
if pca_parameter is not None:
    pca = PCA(n_components=pca_parameter)
data = df
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("LogisticRegression")
end_to_end_process(scaler_paramerter, na_parameter, logistic, logisticParams, data,pca=pca)
print("------------------------------------------------------------")
print("K Nearest Neighbor")
end_to_end_process(scaler_paramerter, na_parameter, knn, knn_params, data,pca=pca)
print("------------------------------------------------------------")
print("RandomForest")
end_to_end_process(scaler_paramerter, na_parameter, randomForest, rf_params, data,pca=pca)
print("------------------------------------------------------------")
print('Bagging')
end_to_end_process(scaler_paramerter, na_parameter, bagging, bg_params, data, pca=pca)
