# S. Ahuja, sudha.ahuja@cern.ch, June 2020

###########
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
import matplotlib
from matplotlib import pyplot as plt
import pickle
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn import linear_model
from scipy.optimize import lsq_linear
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
###########

def train_xgb(df, inputs, output, hyperparams, test_fraction=0.4):
    X_train, X_test, y_train, y_test = train_test_split(df[inputs], df[output], test_size=test_fraction)#random_state=123)
    train = xgb.DMatrix(data=X_train,label=y_train, feature_names=inputs) 
    test = xgb.DMatrix(data=X_test,label=y_test,feature_names=inputs) 
    progress = dict()
    watchlist = [(test, 'eval'), (train, 'train')]
    full = xgb.DMatrix(data=df[inputs],label=df[output],feature_names=inputs) 
    ntrees = hyperparams['n_estimators']
    booster = xgb.train(hyperparams, train, ntrees, watchlist, evals_result=progress) 
    #df['bdt_output'] = booster.predict(full)
    ## retrieve performance metrics
    epochs = len(progress['train']['mphe']) ## mphe, rmse
    x_axis = range(0, epochs)
    # plot loss function
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, progress['train']['mphe'], label='Train')
    ax.plot(x_axis, progress['eval']['mphe'], label='Test')
    ax.legend()
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Error')
    plt.title('XGBoost Loss , '+fe_names[name])
    plt.savefig(plotdir+'xgboost_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
    plt.savefig(plotdir+'xgboost_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

#    xgb_model = xgb.XGBRegressor()
#    regXGB = GridSearchCV(xgb_model,
#                          {'eta': [0.01, 0.1, 0.2, 0.3, 0.4],
#                           'max_depth': [2, 4, 6, 8],
#                           'n_estimators': [10, 20, 50, 100, 200]}, verbose=1)
#    
#    regXGB.fit(X_train, y_train)
#    print("XGB best score: ", regXGB.best_score_)
#    print("XGB best parameters: ", regXGB.best_params_)

#    params = {
        # Parameters that we are going to tune.                                                                           
#        'max_depth':hyperparams['max_depth'],
#        'min_child_weight': hyperparams['min_child_weight'],
#        'eta':.3,
#        'subsample': hyperparams['subsample'],
#        'colsample_bytree': hyperparams['colsample_bytree'],
        # Other parameters                                                                                                    
##        'objective':'reg:pseudohubererror'
 #       'objective':'reg:squarederror'
 #   }

#    gridsearch_params = [
#        (max_depth, min_child_weight,subsample, colsample_bytree)
#        for max_depth in range(4,9)
#        for min_child_weight in range(2,8)
#        for subsample in [i/10. for i in range(7,11)]
#        for colsample_bytree in [i/10. for i in range(7,11)]
#    ]

#    min_mae = float("Inf")
#    best_params = None

#    for max_depth, min_child_weight, subsample, colsample_bytree in gridsearch_params:
#        print("CV with max_depth={}, min_child_weight={}, subsample={}, colsample_bytree={}".format(
#            max_depth,
#            min_child_weight,
#            subsample, 
#            colsample_bytree))
        # Update our parameters
#        params['max_depth'] = max_depth
#        params['min_child_weight'] = min_child_weight
#        params['subsample'] = subsample
#        params['colsample_bytree'] = colsample_bytree

#    for eta in [.3, .2, .1, .05, .01, .005]:
#        print("CV with eta={}".format(eta))
        # We update our parameters
#        params['eta'] = eta

#        cv_results = xgb.cv(
#            params,
#            train,
#            num_boost_round=hyperparams['n_estimators'],
#            seed=42,
#            nfold=5,
#            metrics={'mae'}, ##mphe, mae
#            early_stopping_rounds=10
#        )

        # Update best MAE
#        mean_mphe = cv_results['test-mae-mean'].min() ## test-mphe-mean, test-mae-mean
#        boost_rounds = cv_results['test-mae-mean'].argmin()
#        print("\tMPHE {} for {} rounds".format(mean_mphe, boost_rounds))
#        if mean_mphe < min_mae:
#            min_mae = mean_mphe
#            best_params = (max_depth,min_child_weight,subsample,colsample_bytree)
#            best_params = eta
#    print("Best params: {}, MAE: {}".format(best_params, min_mae))
#    print("Best params: {}, {}, {}, {}, MAE: {}".format(best_params[0], best_params[1], best_params[2], best_params[3], min_mae))
    
    return booster

def rmseff(x, c=0.68):
    """Compute half-width of the shortest interval containing a fraction 'c' of items in a 1D array."""
    x_sorted = np.sort(x, kind="mergesort") 
    m = int(c * len(x)) + 1
    return np.min(x_sorted[m:] - x_sorted[:-m]) / 2.0

def rms(x):
    x_sorted = np.sort(x, kind="mergesort") 
    x_m = np.mean(x)
    x_sqr = np.square(x - x_m)
    x_r = np.mean(x_sqr)
    return np.sqrt(x_r)

###########

fe_names = {}

fe_names[0] = 'Thr'  
fe_names[1] = 'BC+STC'
#fe_names[1] = 'Best Choice'
#fe_names[2] = 'STC'
#fe_names[4] = 'BC course4'

pileup = 'PU200' ## PU0, PU140, PU200
clustering = 'drdefault' ## drdefault, dr015, drOptimal
layercalibration = 'corrPU0' ## derive, corrPU0 
feoptions = ['threshold', 'mixedbcstc'] #'bestchoicedcen', 'supertriggercell', 'bestchoicecourse4'

###########

dir_in1 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_th/'
dir_in2 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_bcdcen_th/'
dir_in3 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_ctc/'
dir_in4 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_mixedfe/'

file_in_eg = {}

file_in_eg[0] = dir_in1+'Thr.hdf5'
file_in_eg[1] = dir_in4+'BCSTC.hdf5'
#file_in_eg[1] = dir_in2+'BCdcen.hdf5'
#file_in_eg[2] = dir_in3+'STC.hdf5'
#file_in_eg[4] = dir_in4+'BCcourse4.hdf5'

###########

dir_out = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/'

file_out_eg = {}

for i, v in enumerate(feoptions):
    file_out_eg[i] = dir_out+'ntuple_'+pileup+'_'+clustering+'_'+feoptions[i]+'_bdt.hdf5'
#file_out_eg[0] = dir_out+'ntuple_'+pileup+'_'+clustering+'_THRESHOLD_bdt.hdf5'
#file_out_eg[1] = dir_out+'ntuple_'+pileup+'_'+clustering+'_BESTCHOICEDCEN_bdt.hdf5'
#file_out_eg[2] = dir_out+'ntuple_'+pileup+'_'+clustering+'_SUPERTRIGGERCELL_bdt.hdf5'
#file_out_eg[1] = dir_out+'ntuple_'+pileup+'_'+clustering+'_MIXEDBCSTC_bdt.hdf5'
#file_out_eg[4] = dir_out+'ntuple_'+pileup+'_'+clustering+'_BESTCHOICECOURSE4_bdt.hdf5'

###########

dir_out_model = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/models/'

file_out_model_c1 = {}
file_out_model_c2 = {}
file_out_model_c3 = {}

for i, v in enumerate(feoptions):
    file_out_model_c1[i] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_'+feoptions[i]+'.pkl'
    file_out_model_c2[i] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_'+feoptions[i]+'.pkl'
    file_out_model_c3[i] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_'+feoptions[i]+'.pkl'

#file_out_model_c1[0] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_threshold.pkl'
#file_out_model_c1[1] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_bestchoicedcen.pkl'
#file_out_model_c1[2] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_supertriggercell.pkl'
#file_out_model_c1[1] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_mixedbcstc.pkl'
#file_out_model_c1[4] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_bestchoicecourse4.pkl'

#file_out_model_c2[0] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_threshold.pkl'
#file_out_model_c2[1] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_bestchoicedcen.pkl'
#file_out_model_c2[2] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_supertriggercell.pkl'
#file_out_model_c2[1] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_mixedbcstc.pkl'
#file_out_model_c2[4] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_bestchoicecourse4.pkl'

#file_out_model_c3[0] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_threshold.pkl'
#file_out_model_c3[1] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_bestchoicedcen.pkl'
#file_out_model_c3[2] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_supertriggercell.pkl'
#file_out_model_c3[1] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_mixedbcstc.pkl'
#file_out_model_c3[4] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_bestchoicecourse4.pkl'

###########

plotdir = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/plots/'

###########

df_eg = {}

for name in file_in_eg:
    df_eg[name]=pd.read_hdf(file_in_eg[name])

###########

print('Selecting clusters')

df_merged_train = {}

df_eg_train = {}

events_total_eg = {}
events_stored_eg = {}

for name in df_eg:

  dfs = []

  # SELECTION

  events_total_eg[name] = np.unique(df_eg[name].reset_index()['event']).shape[0]

  df_eg[name]['cl3d_abseta'] = np.abs(df_eg[name]['cl3d_eta'])

  df_eg_train[name] = df_eg[name]

  sel = df_eg_train[name]['genpart_pt'] > 10
  df_eg_train[name] = df_eg_train[name][sel]
  
  sel = np.abs(df_eg_train[name]['genpart_eta']) > 1.6
  df_eg_train[name] = df_eg_train[name][sel]
  
  sel = np.abs(df_eg_train[name]['genpart_eta']) < 2.9
  df_eg_train[name] = df_eg_train[name][sel]
  
  sel = df_eg_train[name]['best_match'] == True
  df_eg_train[name] = df_eg_train[name][sel]

#  sel = df_eg_train[name]['cl3d_pt'] > 10
#  df_eg_train[name] = df_eg_train[name][sel]

  events_stored_eg[name] = np.unique(df_eg_train[name].reset_index()['event']).shape[0]

print(' ')

###########

## Correcting layer pt with bounded least square
# Training calibration 0 
print('Training calibration for layer pT with lsq_linear')

model_c1 = {}
coefflsq = [1.0 for i in range(14)]

for name in df_eg_train:

    layerpt = df_eg_train[name]['layer']
    cllayerpt = [[0 for col in range(14)] for row in range(len(layerpt))] ##only em layers
    cl3d_layerptsum = []

    for l in range(len(layerpt)):
        layerptSum = 0
        for m in range((len(layerpt.iloc[l]))):
            if(m>0 and m<15):  ## skipping the first layer
                cllayerpt[l][m-1]=layerpt.iloc[l][m]
                layerptSum += cllayerpt[l][m-1]
        cl3d_layerptsum.append(layerptSum)

    df_eg_train[name]['cl3d_layerptsum'] = cl3d_layerptsum
    df_eg_train[name]['cl3d_response_Uncorr'] = df_eg_train[name]['cl3d_layerptsum']/df_eg_train[name]['genpart_pt']

    ## uncorrected 
    mean_Unc_reso = np.mean((df_eg_train[name]['cl3d_pt'])/(df_eg_train[name]['genpart_pt']))
    meanBounds = 1/mean_Unc_reso

    ## layer coefficients
    if(layercalibration == 'derive'):
        blsqregr=lsq_linear(list(cllayerpt), (df_eg_train[name]['genpart_pt']), bounds = (0.5,2.0), method='bvls', lsmr_tol='auto', verbose=1)
        #blsqregr=lsq_linear(list(cllayerpt), (df_eg_train[name]['genpart_pt']), bounds = ((meanBounds)/2.0,(meanBounds)*2), method='bvls', lsmr_tol='auto', verbose=1)
        coefflsq=blsqregr.x
        with open('coefflsq_'+pileup+'_'+clustering+'.txt', 'a') as f:
            print(*coefflsq, sep = ", ",end="\n",file=f) 

    if(layercalibration == 'corrPU0'): 
        with open('coefflsq_PU0_'+clustering+'.txt', 'r') as f:
            for count, line in enumerate(f):
                if(count == name):
                    listcoeff = list(line.split(','))
                    coefflsq = [float(item) for item in listcoeff]

    ## Corrected Pt
    ClPtCorrAll_blsq = {}
    for j in range(len(cllayerpt)):
        ClPtCorr_blsq = 0
        sumlpt = 0
        for k in range(len(cllayerpt[j])):
            sumlpt = sumlpt+cllayerpt[j][k]
            corrlPt_blsq=coefflsq[k]*cllayerpt[j][k]
            ClPtCorr_blsq=ClPtCorr_blsq+corrlPt_blsq
        ClPtCorrAll_blsq[j]=ClPtCorr_blsq
    df_eg_train[name]['cl3d_pt_c0']=list(ClPtCorrAll_blsq.values())
    df_eg_train[name]['cl3d_response_c0'] = df_eg_train[name].cl3d_pt_c0 / df_eg_train[name].genpart_pt

###########

#Defining inputs and targets 

features = ['cl3d_abseta', 'cl3d_n',
  'cl3d_showerlength', 'cl3d_coreshowerlength', 
  'cl3d_firstlayer', 'cl3d_maxlayer', 
  'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean',
  'cl3d_hoe', 'cl3d_meanz', 
  'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 
  'cl3d_ntc67', 'cl3d_ntc90']

for name in df_eg_train:

  df_eg_train[name]['targetAdd'] = np.abs(df_eg_train[name].genpart_pt - df_eg_train[name].cl3d_pt_c0)
  df_eg_train[name]['targetMul'] = df_eg_train[name].genpart_pt/df_eg_train[name].cl3d_pt_c0

###########

# Training calibration 1 
print('Training eta calibration with LinearRegression')

for name in df_eg_train:

  input_c1 = df_eg_train[name][['cl3d_abseta']] 
  target_c1 = df_eg_train[name]['targetAdd']
  model_c1[name] = LinearRegression().fit(input_c1, target_c1)

  df_eg_train[name]['cl3d_c1'] = model_c1[name].predict(df_eg_train[name][['cl3d_abseta']]) 
  df_eg_train[name]['cl3d_pt_c1'] = df_eg_train[name]['cl3d_pt_c0']-(((model_c1[name].coef_)*df_eg_train[name]['cl3d_abseta'])+(model_c1[name].intercept_))
  df_eg_train[name]['cl3d_response_c1'] = df_eg_train[name].cl3d_pt_c1 / df_eg_train[name].genpart_pt

for name in df_eg_train:

  with open(file_out_model_c1[name], 'wb') as f:
    pickle.dump(model_c1[name], f)

###########

# Training calibration 2
print('Training eta calibration with GradientBoostingRegressor')

model_c2_huber = {}
model_c2_ls = {}
model_c2_crossVal_huber = {}
model_c2_crossVal_ls = {}
input_c2 = {}
target_c2 = {} 
GBR_feature_importance_huber = [[None for _ in range(len(features))] for _ in range(len(df_eg_train))]
GBR_feature_importance_ls = [[None for _ in range(len(features))] for _ in range(len(df_eg_train))]
  
for name in df_eg_train:

  input_c2 = df_eg_train[name][features]
  targetMul_c2 = df_eg_train[name]['targetMul']
  targetAdd_c2 = df_eg_train[name]['targetAdd']
  print("Starting GBR for FE option: ", name)

  ##cross-validation and gridsearch
  X_train, X_test, y_train, y_test = train_test_split(input_c2, targetMul_c2, test_size=0.4, random_state=0)

#  param_grid={'n_estimators':[10, 25, 50, 100, 200], 
#              'learning_rate': [0.1, 0.05, 0.02, 0.01], 
#              'max_depth':[2, 4, 6, 8], 
#              'min_samples_leaf':[3,5,9,17], 
#              'max_features':[1.0, 0.7, 0.5, 0.3, 0.1],
#              'loss':['ls']
#              }
#  estimator = GradientBoostingRegressor()
#  regressor = GridSearchCV(estimator=estimator, param_grid=param_grid)
#  regressor.fit(X_train, y_train)
#  best_est = regressor.best_estimator_
#  print("Best Estimator learned through GridSearch")
#  print(regressor.best_estimator_)
  ## GBR best estimator parameters are as follows: 
#  print("Best Estimator Parameters") 
#  print("---------------------------")
#  print( "n_estimators: %d" %best_est.n_estimators)
#  print("max_depth: %d" %best_est.max_depth)
#  print("Learning Rate: %.1f" %best_est.learning_rate)
#  print("min_samples_leaf: %d" %best_est.min_samples_leaf)
#  print("max_features: %.1f" %best_est.max_features)
#  print()
#  print("Train R-squared: %.2f" %best_est.score(X_train,y_train))

  if(name == 0):
      model_c2_crossVal_huber[name] = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, min_samples_leaf = 3, max_features = 1.0, random_state=0, loss='huber').fit(X_train, y_train)
  else:
      model_c2_crossVal_huber[name] = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=8, min_samples_leaf = 3, max_features = 0.7, random_state=0, loss='huber').fit(X_train, y_train)
  if(name == 0):
      model_c2_crossVal_ls[name] = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, min_samples_leaf = 3, max_features = 0.5, random_state=0, loss='ls').fit(X_train, y_train)
  else:
      model_c2_crossVal_ls[name] = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=8, min_samples_leaf = 3, max_features = 0.7, random_state=0, loss='ls').fit(X_train, y_train)

  print("huber GBR train score: ",model_c2_crossVal_huber[name].score(X_train, y_train))
  print("huber GBR test score: ",model_c2_crossVal_huber[name].score(X_test, y_test))
  print("huber GBR MSE: ",mean_squared_error(y_test, model_c2_crossVal_huber[name].predict(X_test)))
  print("ls GBR train score: ",model_c2_crossVal_ls[name].score(X_train, y_train))
  print("ls GBR test score: ",model_c2_crossVal_ls[name].score(X_test, y_test))
  print("ls GBR MSE: ",mean_squared_error(y_test, model_c2_crossVal_ls[name].predict(X_test)))

  if(name == 0):
      model_c2_huber[name] = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, min_samples_leaf = 3, max_features = 1.0, random_state=0, loss='huber').fit(input_c2, targetMul_c2)
  else:
      model_c2_huber[name] = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=8, min_samples_leaf = 3, max_features = 0.1, random_state=0, loss='huber').fit(input_c2, targetMul_c2)      
  if(name == 0):
      model_c2_ls[name] = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0,  min_samples_leaf = 3, max_features = 0.3, loss='ls').fit(input_c2, targetMul_c2)
  else:
      model_c2_ls[name] = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=8, random_state=0,  min_samples_leaf = 3, max_features = 0.7, loss='ls').fit(input_c2, targetMul_c2)

  df_eg_train[name]['cl3d_c2'] = model_c2_huber[name].predict(df_eg_train[name][features])
  df_eg_train[name]['cl3d_pt_c2'] = df_eg_train[name].cl3d_pt_c0 * (df_eg_train[name].cl3d_c2)
  df_eg_train[name]['cl3d_response_c2'] = df_eg_train[name].cl3d_pt_c2 / df_eg_train[name].genpart_pt
  GBR_feature_importance_huber[name] = model_c2_huber[name].feature_importances_

  df_eg_train[name]['cl3d_c2b'] = model_c2_ls[name].predict(df_eg_train[name][features])
  df_eg_train[name]['cl3d_pt_c2b'] = df_eg_train[name].cl3d_pt_c0 * (df_eg_train[name].cl3d_c2b)
  df_eg_train[name]['cl3d_response_c2b'] = df_eg_train[name].cl3d_pt_c2b / df_eg_train[name].genpart_pt
  GBR_feature_importance_ls[name] = model_c2_ls[name].feature_importances_

  ## plotting training deviance
  test_score = np.zeros(50, dtype=np.float64)                                                                                 
  for i, y_pred in enumerate(model_c2_crossVal_huber[name].staged_predict(X_test)):
      test_score[i] = model_c2_crossVal_huber[name].loss_(y_test, y_pred)      
  fig = plt.figure(figsize=(6, 6))
  plt.subplot(1, 1, 1)
  plt.title('Deviance '+fe_names[name])
  plt.plot(np.arange(50), model_c2_crossVal_huber[name].train_score_, 'b-', label='Training Set Deviance')
  plt.plot(np.arange(50), test_score, 'r-', label='Test Set Deviance')
  plt.legend(loc='upper right')
  plt.xlabel('Boosting Iterations')
  plt.ylabel('Deviance')
  fig.tight_layout()
  plt.savefig(plotdir+'GBR_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')         
  plt.savefig(plotdir+'GBR_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')         

  test_score0 = np.zeros(100, dtype=np.float64)                                                                                 
  test_score1 = np.zeros(20, dtype=np.float64)      
  if(name == 0):                                                                           
      for i, y_pred in enumerate(model_c2_crossVal_ls[0].staged_predict(X_test)):
          test_score0[i] = model_c2_crossVal_ls[0].loss_(y_test, y_pred)      
  else:
      for i, y_pred in enumerate(model_c2_crossVal_ls[1].staged_predict(X_test)):
          test_score1[i] = model_c2_crossVal_ls[1].loss_(y_test, y_pred)      
  fig = plt.figure(figsize=(6, 6))
  plt.subplot(1, 1, 1)
  plt.title('Deviance '+fe_names[name])
  if(name == 0):
      plt.plot(np.arange(100), model_c2_crossVal_ls[0].train_score_, 'b-', label='Training Set Deviance')
      plt.plot(np.arange(100), test_score0, 'r-', label='Test Set Deviance')
  else:
      plt.plot(np.arange(20), model_c2_crossVal_ls[1].train_score_, 'b-', label='Training Set Deviance')
      plt.plot(np.arange(20), test_score1, 'r-', label='Test Set Deviance')
  plt.legend(loc='upper right')
  plt.xlabel('Boosting Iterations')
  plt.ylabel('Deviance')
  fig.tight_layout()
  plt.savefig(plotdir+'GBR_training_deviance_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')         
  plt.savefig(plotdir+'GBR_training_deviance_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')         

for name in df_eg_train:

    with open(file_out_model_c2[name], 'wb') as f:
        pickle.dump(model_c2_huber[name], f)

###########

# Training calibration 2 (xgboost)
print('Training eta calibration with xgboost')

param = {}

param['nthread']          	= 10  # limit number of threads
param['silent'] 			      = True
param['n_estimators'] 			    = 100  # number of trees to make
#param['objective']   		    = 'reg:squarederror' #'squarederror','pseudohubererror' # objective function
param['objective']   		    = 'reg:pseudohubererror' #'squarederror','pseudohubererror' # objective function
param['eval_metric']             = 'mphe' ## default for reg:pseudohubererror

model_c3 = {}
model_c3_train = {}
 
for name in df_eg_train:

  print("Starting xgboost for FE option: ", name)

  if(name == 0):
      param['eta']              	= 0.1 # learning rate
      param['max_depth']        	= 5  # maximum depth of a tree
      param['subsample']        	= 0.9 # fraction of events to train tree on
      param['colsample_bytree'] 	= 0.9 # fraction of features to train tree on
      param['min_child_weight']  = 2
      param['n_estimators'] = 200
  else:
      param['eta']              	= 0.3 # learning rate
      param['max_depth']        	= 5  # maximum depth of a tree
      param['subsample']        	= 1.0 # fraction of events to train tree on
      param['colsample_bytree'] 	= 0.9 # fraction of features to train tree on
      param['min_child_weight']  = 2
      param['n_estimators'] = 100

  print("Hyperparameters xgboost: ", param)
  output = 'targetMul' 
  model_c3_train[name] = train_xgb(df_eg_train[name], features, output, param, test_fraction=0.4)

  full = xgb.DMatrix(data=df_eg_train[name][features], label=df_eg_train[name][output], feature_names=features)
  model_c3[name] = xgb.train(param, full, num_boost_round=param['n_estimators'])
  df_eg_train[name]['cl3d_c3'] = model_c3[name].predict(full)
  df_eg_train[name]['cl3d_pt_c3'] = df_eg_train[name].cl3d_pt_c0 * (df_eg_train[name].cl3d_c3)
  df_eg_train[name]['cl3d_response_c3'] = df_eg_train[name].cl3d_pt_c3 / df_eg_train[name].genpart_pt

for name in df_eg_train:

  with open(file_out_model_c3[name], 'wb') as f:
    pickle.dump(model_c3[name], f)

print(' ')

###########

# Results

###########

# Save files

for name in df_eg:

  store_eg = pd.HDFStore(file_out_eg[name], mode='w')
  store_eg['df_eg_PU200'] = df_eg_train[name]
  store_eg.close()

###########

# PLOTTING

colors = {}
colors[0] = 'blue'
colors[1] = 'red'
colors[2] = 'olive'
colors[3] = 'orange'
colors[4] = 'fuchsia'

legends = {}
legends[0] = 'Threshold 1.35 mipT'
legends[0] = 'BC DCentral '
legends[1] = 'STC4+16' 
legends[2] = 'Mixed BC + STC'
legends[3] = 'BC Coarse 2x2 TC'

## FEATURE IMPORTANCES GBR

matplotlib.rcParams.update({'font.size': 16})

for name in df_eg_train:

  plt.figure(figsize=(15,10))
  plt.rcdefaults()
  fig, ax = plt.subplots()
  y_pos = np.arange(len(features))
  ax.barh(y_pos, GBR_feature_importance_huber[name], align='center')
  for index, value in enumerate(GBR_feature_importance_huber[name]):
    ax.text(value+.005, index+.25, str(value), color='black')#, fontweight='bold')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(features)
  ax.invert_yaxis()  # labels read top-to-bottom
  ax.set_xlabel('Feature Importance')
  ax.set_title('Gradient Boosting Regressor (loss=huber)')
  plt.subplots_adjust(left=0.35, right=0.80, top=0.85, bottom=0.2)
  plt.savefig(plotdir+'GBR_bdt_importances_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
  plt.savefig(plotdir+'GBR_bdt_importances_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

for name in df_eg_train:

  plt.figure(figsize=(15,10))
  plt.rcdefaults()
  fig, ax = plt.subplots()
  y_pos = np.arange(len(features))
  ax.barh(y_pos, GBR_feature_importance_ls[name], align='center')
  for index, value in enumerate(GBR_feature_importance_ls[name]):
    ax.text(value+.005, index+.25, str(value), color='black')#, fontweight='bold')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(features)
  ax.invert_yaxis()  # labels read top-to-bottom
  ax.set_xlabel('Feature Importance')
  ax.set_title('Gradient Boosting Regressor (loss=ls)')
  plt.subplots_adjust(left=0.35, right=0.80, top=0.85, bottom=0.2)
  plt.savefig(plotdir+'GBR_bdt_importances_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
  plt.savefig(plotdir+'GBR_bdt_importances_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

## FEATURE IMPORTANCES XGBOOST

matplotlib.rcParams.update({'font.size': 16})

for name in df_eg_train:
  plt.figure(figsize=(15,10))
  xgb.plot_importance(model_c3[name], grid=False, importance_type='gain',lw=2)
  plt.title('XGBOOST Regressor')
  plt.subplots_adjust(left=0.50, right=0.85, top=0.9, bottom=0.2)
  plt.savefig(plotdir+'XGBOOST_bdt_importances_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
  plt.savefig(plotdir+'XGBOOST_bdt_importances_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

## CALIBRATION COMPARISON

for name in df_eg_train:

  plt.figure(figsize=(15,10))
  plt.title(fe_names[name])
  plt.hist((df_eg_train[name].cl3d_response_Uncorr), bins=np.arange(0.0, 1.4, 0.01), label = 'Uncorrected', histtype = 'step')
  plt.hist((df_eg_train[name].cl3d_response_c0), bins=np.arange(0.0, 1.4, 0.01), label = 'Layer weights (C)', histtype = 'step')
  plt.hist((df_eg_train[name].cl3d_response_c1), bins=np.arange(0.0, 1.4, 0.01), label = 'C + LR', histtype = 'step')
  plt.hist((df_eg_train[name].cl3d_response_c2), bins=np.arange(0.0, 1.4, 0.01), label = 'C + GBR (huber)', histtype = 'step')
  plt.hist((df_eg_train[name].cl3d_response_c2b), bins=np.arange(0.0, 1.4, 0.01), label = 'C + GBR (ls)', histtype = 'step')
  plt.hist((df_eg_train[name].cl3d_response_c3),  bins=np.arange(0.0, 1.4, 0.01),  label = 'C + xgboost', histtype = 'step')
  plt.xlabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
  plt.xlim(0.2,1.4)
  plt.legend(frameon=False)    
  plt.grid(False)
  plt.subplots_adjust(left=0.28, right=0.85, top=0.9, bottom=0.1)
  plt.savefig(plotdir+'calibration_response_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
  plt.savefig(plotdir+'calibration_response_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

#Pt and eta bin response

for name in df_eg_train:

    df_eg_train[name]['bineta'] = ((np.abs(df_eg_train[name]['genpart_eta']) - 1.6)/0.13).astype('int32')
    df_eg_train[name]['binpt'] = ((df_eg_train[name]['genpart_pt']- 10.0)/10.0).astype('int32')
    df_mean_eta = df_eg_train[name].groupby(['bineta']).mean()
    df_mean_pt = df_eg_train[name].groupby(['binpt']).mean()
    df_effrms_eta = df_eg_train[name].groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_Uncorr))
    df_effrms_pt = df_eg_train[name].groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_Uncorr))
    df_effrms_etaC1 = df_eg_train[name].groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c1))
    df_effrms_ptC1 = df_eg_train[name].groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c1)) 
    df_effrms_etaC2 = df_eg_train[name].groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c2))
    df_effrms_ptC2 = df_eg_train[name].groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c2)) 
    df_effrms_etaC2b = df_eg_train[name].groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c2b))
    df_effrms_ptC2b = df_eg_train[name].groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c2b)) 
    df_effrms_etaC3 = df_eg_train[name].groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c3))
    df_effrms_ptC3 = df_eg_train[name].groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c3)) 
    df_rms_eta = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_Uncorr))
    df_rms_pt = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_Uncorr))
    df_rms_etaC1 = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_c1))
    df_rms_ptC1 = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_c1))
    df_rms_etaC2 = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_c2))
    df_rms_ptC2 = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_c2))
    df_rms_etaC3 = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_c3))
    df_rms_ptC3 = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_c3))
    if(name==0):
        fig = plt.figure(num='performance',figsize=(32,40))
        plt.title(pileup+'_'+clustering)
    plt.figure(num='performance')
    plt.subplot(541)
    plt.title('Uncorrected')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(542)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(543) 
    plt.errorbar((df_mean_eta.cl3d_abseta), df_effrms_eta/df_mean_eta.cl3d_response_Uncorr, linestyle='-', marker='o',  label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(544) 
    plt.errorbar(np.abs(df_mean_pt.genpart_pt), df_effrms_pt/df_mean_pt.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(545)
    plt.title('Linear Regression')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(546)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(547)
    plt.errorbar((df_mean_eta.cl3d_abseta),df_effrms_etaC1/df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(548)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC1/df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(549)
    plt.title('Gradient Boosting Regressor (loss=huber)')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,10)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,11)
    plt.errorbar((df_mean_eta.cl3d_abseta),df_effrms_etaC2/df_mean_eta.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,12)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC2/df_mean_pt.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(5,4,13)
    plt.title('Gradient Boosting Regressor (loss=ls)')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,14)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,15)
    plt.errorbar((df_mean_eta.cl3d_abseta),df_effrms_etaC2b/df_mean_eta.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,16)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC2b/df_mean_pt.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(5,4,17)
    plt.title('XGBOOST Regression')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,18)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,19)
    plt.errorbar((df_mean_eta.cl3d_abseta),df_effrms_etaC3/df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(5,4,20)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC3/df_mean_pt.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
#    plt.subplots_adjust(left=0.28, right=0.85, top=0.9, bottom=0.1)
    plt.savefig(plotdir+'calibration_performancesummary_'+pileup+'_'+clustering+'.png')
    plt.savefig(plotdir+'calibration_performancesummary_'+pileup+'_'+clustering+'.pdf')

###########

