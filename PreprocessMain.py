import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Dataset_1.csv')
r_scaler = RobustScaler()

def ScalingData(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return df_scaled

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
            print(df[c].unique())
    elif work=="drop":
        df = df.dropna(subset=col)
    elif work=="stratify":
        print("비율에 맞게 나눠주는 코드")
    return df

print('--------------------------------------------------------------------------')
print(df['Geography'].mode(),'modeVal')
print(df.info())#info 출력을 통해 대략적 정보를 얻음.

foundNan(df)#nan이 존재, Endcoding이 불가능한 값은 최빈값으로 대체 (카테고리컬 값은)



df=df.drop(['Unnamed: 0','Surname'],axis=1)
df=handleNa(['Geography','Gender'],df,work="mode")#mode로 categorical 제거...
foundNan(df)

for c in df:
    print("col %s : %d"%(c,len(df[c].unique())))
    print('----------------------------------------')
print('----------------------------------------')
df = columnEncoding(['Geography','Gender'],df)#Encoding을 통해 String vlaue를 제거

df = handleNa(df.columns,df,work="mean") # 나머지 value들 mean으로 fillna
foundNan(df)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
X = pd.DataFrame(ScalingData(X,r_scaler),columns=X.columns)

from sklearn.model_selection import train_test_split

X, X_eval, Y, Y_eval = train_test_split(X,Y,stratify=Y,random_state=8, test_size=.25)
print(X.shape)
print(Y.shape)
print(X_eval.shape)
print(Y_eval.shape)
#-------------------------------------------------------------
#                       Process
#-------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def confirmAccuracy(gt, result):
    correct=0
    for i in range(len(gt)):
        if(gt[i]==result[i]):
            correct+=1
    return correct/len(gt)*100

def predict(X_train, Y_train, X_ev, Y_ev, model, param, cv=5):
    print(param, 'params')
    gridSearchModel = GridSearchCV(model, param_grid=param, cv=cv, refit=True)
    gridSearchModel.fit(X_train,Y_train)
    print("Best Parameter: [%s]"%(gridSearchModel.best_params_))
    print("Best Score(test Set): [%f]"%(gridSearchModel.best_score_))
    fitted_model = gridSearchModel.best_estimator_
    predict = fitted_model.predict(X_train)
    print("Score with TrainSet: [%f]"%(confirmAccuracy(Y_train.values,predict)))
    predict = fitted_model.predict(X_ev)
    print("Score with EvalSet: [%f]"%(confirmAccuracy(Y_ev.values, predict)))

logisticParams ={
    'penalty':['l1','l2','elasticnet'],
    'dual':[True,False],
    'C':[0.25,0.5,1.0,2.0,3.0,4.0],
    'fit_intercept':[True,False]
}

knn_params={'n_neighbors':np.arange(1,5),
        'weights':['uniform','distance'],}

rf_params = {
    'n_estimators':np.arange(4, 30, 2),
    'criterion':['gini','entropy'],
    'max_depth':[None,1,2,3]
}

print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("LogisticRegression")
logistic = LogisticRegression()
predict(X,Y,X_eval,Y_eval,logistic, logisticParams)
print("------------------------------------------------------------")
print("K Nearest Neighbor")
knn = KNeighborsClassifier()
predict(X,Y,X_eval,Y_eval,knn, knn_params)
print("------------------------------------------------------------")
print("RandomForest")
randomForest = RandomForestClassifier()
predict(X,Y,X_eval,Y_eval,randomForest, rf_params)
print("------------------------------------------------------------")