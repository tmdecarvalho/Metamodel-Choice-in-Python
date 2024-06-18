# reproduction of the case study in de Carvalho et al MDM 2021 publication in Python
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import ensemble
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import keras
import os
#from keras.layers import Input, Dense


import sklearn.ensemble


# printing options
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}) # prints with 2 decimals ?
np.set_printoptions(precision=3)
pd.options.display.float_format = '{:.3f}'.format  # prints with 2 decimals

# ensure reproducibility (NB: different libraries have different random number generators!!!!!)
np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Setting the environment variable to enforce deterministic operations in TF


# models
# OLS (verification),  RF, xgboost, GP?, ANNs
emulatornames = ["LM","RF","xgboost", "GP","ANN"]
accuracynames = ["predmean CV","PAE CV","MAE CV","MSE CV","RMSE CV",
                 "predmean all","PAE all ","MAE all","MSE all","RMSE all",
                 "hyperp1","hyperp2","hyperp3","hyperp4","hyperp5","hyperp6"]


errorstats =  pd.DataFrame(np.zeros((len(emulatornames), len(accuracynames))))
errorstats.columns = accuracynames



#  rownames(errorstats) <- emulatornames


# select model outcomes

modeloutcomes = 7

totalmodelparams = np.arange(22)  # total model params
# NB: np.array(a) just creates a scalar!

# if outcome is cost => utilities do not enter metamodel and costs of competing treatment
# if outcome is qaly => costs do not enter metamodel and also not qaly of the competing treatment

# compared to R i subtract everything by 1 since python starts counting at 0.

# outcomes in MDM publication: 2,3,7,8
if modeloutcomes  in [0, 2]:
    modelparams = np.delete(totalmodelparams, [10, 12, 13, 14, 15, 16, 17, 18, 20])
elif modeloutcomes in [5, 7]:
    modelparams = np.delete(totalmodelparams, [14, 19, 20, 21])
elif modeloutcomes in [1, 3]:
    modelparams = np.delete(totalmodelparams, [10, 11, 13, 14, 15, 16, 17, 18, 19])
elif modeloutcomes in [6, 8]:
    modelparams = np.delete(totalmodelparams, [11, 13, 19, 20, 21])

print(modelparams)

modelparams = modelparams.tolist()


# train and test data
exp_des_valid = pd.read_excel("LC_PSA_dataset.xlsx",sheet_name=2, dtype=float)
out_valid = pd.read_excel("LC_PSA_dataset.xlsx",sheet_name=1, dtype=float)
exp_des_train = pd.read_excel("LC_PSA_dataset.xlsx",sheet_name=4, dtype=float)
out_train = pd.read_excel("LC_PSA_dataset.xlsx",sheet_name=3, dtype=float)


# cross-valid

nparams = len(modelparams) # parameters included in metamodel,
n_params =  np.arange(nparams)
#print(n_params)
sample_param = 10
n_expdes = len(totalmodelparams)* sample_param
cvfold = int(n_expdes/sample_param)
print(cvfold)

number_cv = np.arange( round(n_expdes/cvfold) )
cvobs =  np.array(n_expdes)

row_list = list(range(0, len(out_train)))



H = exp_des_train.iloc[:,modelparams]
#print("this is the X matrix")

Z = pd.DataFrame(columns=H.columns, index=range(n_expdes), dtype=float)
out_mod = out_train.iloc[:,modeloutcomes]
H_pred = exp_des_valid.iloc[:,modelparams]
Z_pred = pd.DataFrame(columns=H.columns, index=range(len(H_pred.iloc[:,1])),dtype=float)
out_mod_pred = out_valid.iloc[:,modeloutcomes]


# standardize data
#y_stand = pd.DataFrame(index=range(len(out_mod)),columns=range(0))
y_stand =  (out_mod - out_mod.mean()) / out_mod.std()

y_range = out_mod.max() - out_mod.min()
y_range2 = out_mod_pred.max() - out_mod_pred.min()

for i,p in enumerate(n_params):
    #print(i)
    Z.iloc[:,i] =  (H.iloc[:,i] - H.iloc[:,i].mean() ) / H.iloc[:,i].std()
    #Z_pred.iloc[:,i] = ( H_pred.iloc[:,i] - H_pred.iloc[:,i].mean() ) / H_pred.iloc[:,i].std()
    Z_pred.iloc[:, i] = (H_pred.iloc[:, i] - H.iloc[:, i].mean()) / H.iloc[:, i].std()

# hyperparameters
hyperparams_rf = [nparams, round(nparams/1.5),round(nparams/2),round(nparams/3)]
hyperparams_bo1 = [0.01,0.05,0.25,0.01,0.05,0.25]
hyperparams_bo2 = [0.6,0.6,0.6,0.8,0.8,0.8]
hyperparams_nn = [nparams, round(nparams/1.5),round(nparams/2),round(nparams/3)]

#other hyperparsm
ntrees = 250 # trees for rf


# save predictions
#  predict_ols = np.arange(range(len(out_mod)))  --> this is WRONG!!!!
predict_ols = np.array(range(len(out_mod)),dtype=float)
predict_gp = np.array(range(len(out_mod)),dtype=float)
predict_rf = pd.DataFrame(columns=hyperparams_rf, index=range(len(out_mod)))
predict_bo = pd.DataFrame(columns=hyperparams_bo1, index=range(len(out_mod)))
predict_nn = pd.DataFrame(columns=hyperparams_nn, index=range(len(out_mod)))
#predict_ols = pd.DataFrame(index=range(len(out_mod)),columns=range(0))



# Linear Model: it should give the same prediction error as in R  ****************************************************

for cv in number_cv:

    test_rows = row_list[(cvfold * cv):(cvfold * (cv+1) )]

    if cv > 0:
        train_rows = row_list[:(cvfold * cv)] + row_list[(cvfold * (cv+1)):]
    elif cv == 0:
        train_rows = row_list[(cvfold * (cv+1)):]

    y = y_stand.iloc[train_rows]
    X = Z.iloc[train_rows,]

    OLS_model = LinearRegression()
    OLS_model.fit(X,y)
    #print(OLS_model.predict(Z.iloc[test_rows,])  * out_mod.std()  +  out_mod.mean())
    test1 = out_mod.mean()
    test2 = out_mod.std()
    test3 = OLS_model.predict(Z.iloc[test_rows,:])
    predict_ols[test_rows]  = OLS_model.predict(Z.iloc[test_rows,:])  * out_mod.std()  +  out_mod.mean()


df_aux  = predict_ols - out_mod
df_aux2 = (predict_ols  - out_mod).abs()


errorstats.iloc[0, 0] =   predict_ols.mean()
errorstats.iloc[0, 1] =   (df_aux2/ out_mod.abs() * 100).mean()
errorstats.iloc[0, 2] =   df_aux2.mean()
errorstats.iloc[0, 3] =   np.square(df_aux).mean()
errorstats.iloc[0, 4] =   ( np.sqrt ( np.square(df_aux) ) / y_range ).mean()

OLS_model2 = LinearRegression()
OLS_model2.fit(Z, y_stand)
predict_ols2 = OLS_model2.predict(Z_pred) * out_mod.std() + out_mod.mean()


df_aux  = predict_ols2 - out_mod_pred
df_aux2 = (predict_ols2  - out_mod_pred).abs()
errorstats.iloc[0, 5] =  predict_ols2.mean()
errorstats.iloc[0, 6] =   (df_aux2/ out_mod_pred.abs() * 100).mean()
errorstats.iloc[0, 7] =   df_aux2.mean()
errorstats.iloc[0, 8] =   np.square(df_aux).mean()
errorstats.iloc[0, 9] =   ( np.sqrt ( np.square(df_aux) ) / y_range2 ).mean()


# RF  : Random Forest metamodel ***************************************************************************************

for cv in number_cv:

    test_rows = row_list[(cvfold * cv):(cvfold * (cv+1) )]

    if cv > 0:
        train_rows = row_list[:(cvfold * cv)] + row_list[(cvfold * (cv+1)):]
    elif cv == 0:
        train_rows = row_list[(cvfold * (cv+1)):]

    y = y_stand.iloc[train_rows]
    X = Z.iloc[train_rows,]

    for param,i in enumerate(hyperparams_rf):
      rf_model = ensemble.RandomForestRegressor(max_features=hyperparams_rf[param],n_estimators=ntrees)
      rf_model.fit(X, y)
      predict_rf.iloc[test_rows,param]  = rf_model.predict(Z.iloc[test_rows,])  * out_mod.std()  +  out_mod.mean()



mean_square_rf = np.arange(len(hyperparams_rf))
for param,i in enumerate(hyperparams_rf):
    df_auxx = (predict_rf.iloc[:,param]  - out_mod)
    df_auxx = np.array(df_auxx,dtype=float)
    mean_square_rf[param]  =  (np.sqrt(np.square(df_auxx))).mean() #).mean()

index = np.argmin(mean_square_rf)


df_aux  = predict_rf.iloc[:,index] - out_mod
df_aux = np.array(df_aux,dtype=float)
df_aux2 = (predict_rf.iloc[:,index]  - out_mod).abs()
#print(sum(df_aux2)/220)
errorstats.iloc[1, 0] =   predict_rf.iloc[:,index].mean()
errorstats.iloc[1, 1] =   (df_aux2/ out_mod.abs() * 100).mean()
errorstats.iloc[1, 2] =   df_aux2.mean()
errorstats.iloc[1, 3] =   np.square(df_aux).mean()
errorstats.iloc[1, 4] =   ( np.sqrt ( np.square(df_aux) ) / y_range ).mean()

rf_model2 = ensemble.RandomForestRegressor(max_features=hyperparams_rf[index],n_estimators=ntrees)
rf_model2.fit(Z, y_stand)
predict_rf2 =   rf_model2.predict(Z_pred) * out_mod.std() + out_mod.mean()


df_aux  = predict_rf2 - out_mod_pred
df_aux2 = (predict_rf2  - out_mod_pred).abs()
errorstats.iloc[1, 5] =  predict_rf2.mean()
errorstats.iloc[1, 6] =   (df_aux2/ out_mod_pred.abs() * 100).mean()
errorstats.iloc[1, 7] =   df_aux2.mean()
errorstats.iloc[1, 8] =   np.square(df_aux).mean()
errorstats.iloc[1, 9] =   ( np.sqrt ( np.square(df_aux) ) / y_range2 ).mean()

for param,i in enumerate(hyperparams_rf):
    errorstats.iloc[1, param+10] = mean_square_rf[param]

# Gradient boosting   ************************************************************************************************

for cv in number_cv:

    test_rows = row_list[(cvfold * cv):(cvfold * (cv+1) )]

    if cv > 0:
        train_rows = row_list[:(cvfold * cv)] + row_list[(cvfold * (cv+1)):]
    elif cv == 0:
        train_rows = row_list[(cvfold * (cv+1)):]

    y = y_stand.iloc[train_rows]
    X = Z.iloc[train_rows,]

    for param,i in enumerate(hyperparams_bo1):
      bo_model = ensemble.GradientBoostingRegressor(learning_rate=hyperparams_bo1[param],subsample =hyperparams_bo2[param],
                                                    n_estimators=ntrees)
      bo_model.fit(X, y)
      predict_bo.iloc[test_rows,param]  = bo_model.predict(Z.iloc[test_rows,])  * out_mod.std()  +  out_mod.mean()



mean_square_bo = np.arange(len(hyperparams_bo1))
for param,i in enumerate(hyperparams_bo1):
    df_auxx = (predict_bo.iloc[:,param]  - out_mod)
    df_auxx = np.array(df_auxx,dtype=float)
    mean_square_bo[param]  =  (np.sqrt(np.square(df_auxx))).mean() #).mean()

index = np.argmin(mean_square_bo)

df_aux  = predict_bo.iloc[:,index] - out_mod
df_aux = np.array(df_aux,dtype=float)
df_aux2 = (predict_bo.iloc[:,index]  - out_mod).abs()

errorstats.iloc[2, 0] =   predict_bo.iloc[:,index].mean()
errorstats.iloc[2, 1] =   (df_aux2/ out_mod.abs() * 100).mean()
errorstats.iloc[2, 2] =   df_aux2.mean()
errorstats.iloc[2, 3] =   np.square(df_aux).mean()
errorstats.iloc[2, 4] =   ( np.sqrt ( np.square(df_aux) ) / y_range ).mean()

bo_model2 = ensemble.GradientBoostingRegressor(learning_rate=hyperparams_bo1[index],subsample =hyperparams_bo2[index],
                                                n_estimators=ntrees)
bo_model2.fit(Z, y_stand)
predict_bo2 =   bo_model2.predict(Z_pred) * out_mod.std() + out_mod.mean()


df_aux  = predict_bo2 - out_mod_pred
df_aux2 = (predict_bo2  - out_mod_pred).abs()
errorstats.iloc[2, 5] =  predict_bo2.mean()
errorstats.iloc[2, 6] =   (df_aux2/ out_mod_pred.abs() * 100).mean()
errorstats.iloc[2, 7] =   df_aux2.mean()
errorstats.iloc[2, 8] =   np.square(df_aux).mean()
errorstats.iloc[2, 9] =   ( np.sqrt ( np.square(df_aux) ) / y_range2 ).mean()


for param,i in enumerate(hyperparams_bo1):
     errorstats.iloc[2, param+10] = mean_square_bo[param]


# GAUSSIAN PROCESS regression    **************************************************************************************


for cv in number_cv:

    test_rows = row_list[(cvfold * cv):(cvfold * (cv+1) )]

    if cv > 0:
        train_rows = row_list[:(cvfold * cv)] + row_list[(cvfold * (cv+1)):]
    elif cv == 0:
        train_rows = row_list[(cvfold * (cv+1)):]

    y = y_stand.iloc[train_rows]
    X = Z.iloc[train_rows,]

    kernel = RBF()
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=False)
    gp_model.fit(X, y)
    predict_gp[test_rows]  = gp_model.predict(Z.iloc[test_rows,])  * out_mod.std()  +  out_mod.mean()


df_aux  = predict_gp - out_mod
df_aux2 = (predict_gp  - out_mod).abs()

errorstats.iloc[3, 0] =   predict_gp.mean()
errorstats.iloc[3, 1] =   (df_aux2/ out_mod.abs() * 100).mean()
errorstats.iloc[3, 2] =   df_aux2.mean()
errorstats.iloc[3, 3] =   np.square(df_aux).mean()
errorstats.iloc[3, 4] =   ( np.sqrt ( np.square(df_aux) ) / y_range ).mean()

kernel = RBF()
gp_model2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=False)
gp_model2.fit(Z, y_stand)
predict_gp2 =   gp_model2.predict(Z_pred) * out_mod.std() + out_mod.mean()


df_aux  = predict_gp2 - out_mod_pred
df_aux2 = (predict_gp2  - out_mod_pred).abs()
errorstats.iloc[3, 5] =  predict_gp2.mean()
errorstats.iloc[3, 6] =   (df_aux2/ out_mod_pred.abs() * 100).mean()
errorstats.iloc[3, 7] =   df_aux2.mean()
errorstats.iloc[3, 8] =   np.square(df_aux).mean()
errorstats.iloc[3, 9] =   ( np.sqrt ( np.square(df_aux) ) / y_range2 ).mean()


# ANNs  **************************************************************************************************************
# note: sigmoid activation function used here (for better comparability with the MDM publication), but
# relu would be recommended here.
for cv in number_cv:

    test_rows = row_list[(cvfold * cv):(cvfold * (cv+1) )]

    if cv > 0:
        train_rows = row_list[:(cvfold * cv)] + row_list[(cvfold * (cv+1)):]
    elif cv == 0:
        train_rows = row_list[(cvfold * (cv+1)):]

    my_vector = list(range(198))
    #y = pd.DataFrame(my_vector, columns=['y'])
    y = y_stand.iloc[train_rows]
    X = Z.iloc[train_rows,]

    for param,i in enumerate(hyperparams_nn):

        print(param)
        print(hyperparams_nn[param])

        ann_model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(len(modelparams), 1)),
            tf.keras.layers.Dense(hyperparams_nn[param], activation='sigmoid'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Flatten(),  # Flatten the output !!! otherwise returns xdim by 1 vector!!!!
            tf.keras.layers.Dense(1)
        ])
        ann_model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-2), loss='mean_squared_error')
        ann_model.fit(X, y, epochs=50,batch_size= 75,shuffle=False)
        predictions = ann_model(Z.iloc[test_rows,],training=False)
        predictions = predictions.numpy()  # convert tensor to numpy, note we need to assign it, otherwise it wont work!!!
        predictions = predictions.reshape(-1)  # remove the extra (useless) dimension
        predict_nn.iloc[test_rows,param]  = predictions  * out_mod.std()  +  out_mod.mean()


mean_square_nn = np.arange(len(hyperparams_nn))
for param,i in enumerate(hyperparams_nn):
    df_auxx = (predict_nn.iloc[:,param]  - out_mod)
    df_auxx = np.array(df_auxx,dtype=float)
    mean_square_nn[param]  =  (np.sqrt(np.square(df_auxx))).mean() #).mean()

index = np.argmin(mean_square_nn)

df_aux  = predict_nn.iloc[:,index] - out_mod
df_aux = np.array(df_aux,dtype=float)
df_aux2 = (predict_nn.iloc[:,index]  - out_mod).abs()
#print(sum(df_aux2)/220)
errorstats.iloc[4, 0] =   predict_nn.iloc[:,index].mean()
errorstats.iloc[4, 1] =   (df_aux2/ out_mod.abs() * 100).mean()
errorstats.iloc[4, 2] =   df_aux2.mean()
errorstats.iloc[4, 3] =   np.square(df_aux).mean()
errorstats.iloc[4, 4] =   ( np.sqrt ( np.square(df_aux) ) / y_range ).mean()

ann_model2 = tf.keras.models.Sequential([
    tf.keras.Input(shape=(len(modelparams), 1)),
    tf.keras.layers.Dense(hyperparams_nn[index], activation='sigmoid'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Flatten(),  # Flatten the output !!! otherwise returns xdim by 1 vector!!!!
    tf.keras.layers.Dense(1)
])

ann_model2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss='mean_squared_error')
ann_model2.fit(Z, y_stand,epochs=50,batch_size= 75,shuffle=False)
ann_model2.summary()
predict_nn2 =   ann_model2(Z_pred,training=False) * out_mod.std() + out_mod.mean()
predict_nn2 = predict_nn2.numpy()
predict_nn2 = predict_nn2.reshape(-1)

df_aux  = predict_nn2 - out_mod_pred
df_aux2 = (predict_nn2  - out_mod_pred).abs()
errorstats.iloc[4, 5] =  predict_nn2.mean()
errorstats.iloc[4, 6] =   (df_aux2/ out_mod_pred.abs() * 100).mean()
errorstats.iloc[4, 7] =   df_aux2.mean()
errorstats.iloc[4, 8] =   np.square(df_aux).mean()
errorstats.iloc[4, 9] =   ( np.sqrt ( np.square(df_aux) ) / y_range2 ).mean()

for param,i in enumerate(hyperparams_nn):
    errorstats.iloc[4, param+10] = mean_square_nn[param]


# print results in console
print(errorstats.iloc[0])
print(errorstats.iloc[1])
print(errorstats.iloc[2])
print(errorstats.iloc[3])
print(errorstats.iloc[4])

# print in csv
filenamecsv = "casestudy_mdm.csv"
errorstats.to_csv(filenamecsv,index=True)