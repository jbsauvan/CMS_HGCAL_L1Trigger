# S. Ahuja, sudha.ahuja@cern.ch, June 2020
#### The following script runs different models for performing calibration of the level-1 HGCAL clusters. It uses pre-processed events obtained from the TPG ntuples. The calibration is tested using regression models from Scikit-learn LinearRegression, and Boosted Decision Tree libaries from Scikit-learn GradientBoostingRegressor and XGBoost. The results obtained from each of these libraries are compared in the end to check which library performs better.
#### The current script is tested on a sample of electron events. Before performing calibration using any of the regression libraries (mentitoned above), the cluster layer wise pt is corrected with layer weights obtained using bounded least square method.

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
###########

# FOR PLOTTING

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
### Variable definitions ##
###########
fe_names = {}

fe_names[0] = 'Thr'  
fe_names[1] = 'BC+STC'
#fe_names[1] = 'Best Choice'
#fe_names[2] = 'STC'
#fe_names[4] = 'BC course4'

pileup = 'PU200' ## PU0, PU140, PU200
clustering = 'drdefault' ## drdefault, dr015, drOptimal
layercalibration = 'corrPU0' ## derive (used for PU0 events), corrPU0 (used for PU200 events) 
feoptions = ['threshold', 'mixedbcstc'] #'bestchoicedcen', 'supertriggercell', 'bestchoicecourse4'
tuningGBR = 'False' # set to True if performing hyperparameter scan for GradientBoostingRegressor
tuningXGBOOST = 'True' # set to True if performing hyperparameter scan for XGBOOST
lossfuncdir = 'final_el_drdefault' # huber or ls or final (output directory names used for storing plots 

matplotlib.rcParams.update({'font.size': 16})

###########
dir_in1 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiE_Pt10To100_Eta1p6To2p9_'+pileup+'_thbc_dr0pt1/'
dir_in2 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiE_Pt10To100_Eta1p6To2p9_'+pileup+'_mixedfe_dr0pt1/'

file_in_eg = {}

file_in_eg[0] = dir_in1+'Thr.hdf5'
file_in_eg[1] = dir_in2+'BCSTC.hdf5'
#file_in_eg[0] = dir_in1+'Thrdr50.hdf5'
#file_in_eg[1] = dir_in2+'BCSTCdr050.hdf5'

###########

dir_out = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/'+lossfuncdir+'/'

file_out_eg = {}

for i, v in enumerate(feoptions):
    file_out_eg[i] = dir_out+'ntuple_'+pileup+'_'+clustering+'_'+feoptions[i]+'_bdt.hdf5'

###########

dir_out_model = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/models/'+lossfuncdir+'/'

file_out_model_c1 = {}
file_out_model_c2 = {}
file_out_model_c3 = {}

for i, v in enumerate(feoptions):
    file_out_model_c1[i] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_'+feoptions[i]+'.pkl'
    file_out_model_c2[i] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_'+feoptions[i]+'.pkl'
    file_out_model_c3[i] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_'+feoptions[i]+'.pkl'

###########

plotdir = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/plots/'+lossfuncdir+'/'

###########

df_eg = {}

for name in file_in_eg:
    df_eg[name]=pd.read_hdf(file_in_eg[name])

###########
### Selection of input events ##
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
### Calibration models ##
###########

## Correcting layer pt with bounded least square 
## This is first necessary step and is done for all the calibration models being studied
# Training calibration 0 
print('Training calibration for layer pT with lsq_linear')

model_c1 = {}
coefflsq = [1.0 for i in range(14)]

for name in df_eg_train:

    layerpt = df_eg_train[name]['layer']
    cllayerpt = [[0 for col in range(14)] for row in range(len(layerpt))] ##only em layers
    cl3d_layerptsum = []

    #### considering layer wise pt in the dataframe
    df_eg_train[name] = df_eg_train[name].assign(layer1pt=0.0, layer2pt=0.0, layer3pt=0.0, layer4pt=0.0, layer5pt=0.0, layer6pt=0.0, layer7pt=0.0, layer8pt=0.0, layer9pt=0.0, layer10pt=0.0, layer11pt=0.0, layer12pt=0.0, layer13pt=0.0, layer14pt=0.0)

    for l in range(len(layerpt)):
        layerptSum = 0
        for m in range((len(layerpt.iloc[l]))):
            if(m>0 and m<15):  ## skipping the first layer
                cllayerpt[l][m-1]=layerpt.iloc[l][m]
                layerptSum += cllayerpt[l][m-1]
        cl3d_layerptsum.append(layerptSum)
        df_eg_train[name]['layer1pt'].iloc[l] = float(cllayerpt[l][0])
        df_eg_train[name]['layer2pt'].iloc[l] = float(cllayerpt[l][1])
        df_eg_train[name]['layer3pt'].iloc[l] = float(cllayerpt[l][2])
        df_eg_train[name]['layer4pt'].iloc[l] = float(cllayerpt[l][3])
        df_eg_train[name]['layer5pt'].iloc[l] = float(cllayerpt[l][4])
        df_eg_train[name]['layer6pt'].iloc[l] = float(cllayerpt[l][5])
        df_eg_train[name]['layer7pt'].iloc[l] = float(cllayerpt[l][6])
        df_eg_train[name]['layer8pt'].iloc[l] = float(cllayerpt[l][7])
        df_eg_train[name]['layer9pt'].iloc[l] = float(cllayerpt[l][8])
        df_eg_train[name]['layer10pt'].iloc[l] = float(cllayerpt[l][9])
        df_eg_train[name]['layer11pt'].iloc[l] = float(cllayerpt[l][10])
        df_eg_train[name]['layer12pt'].iloc[l] = float(cllayerpt[l][11])
        df_eg_train[name]['layer13pt'].iloc[l] = float(cllayerpt[l][12])
        df_eg_train[name]['layer14pt'].iloc[l] = float(cllayerpt[l][13])

    df_eg_train[name]['cl3d_layerptsum'] = cl3d_layerptsum
    df_eg_train[name]['cl3d_response_Uncorr'] = df_eg_train[name]['cl3d_layerptsum']/df_eg_train[name]['genpart_pt']

    ## uncorrected 
    mean_Unc_reso = np.mean((df_eg_train[name]['cl3d_pt'])/(df_eg_train[name]['genpart_pt']))
    meanBounds = 1/mean_Unc_reso

    ## layer coefficients ('derive' layer co-efficients first using PU0 events and use the same layer weights for PU200 events)
    if(layercalibration == 'derive'):  ## for the PU0 case
        blsqregr=lsq_linear(list(cllayerpt), (df_eg_train[name]['genpart_pt']), bounds = (0.5,2.0), method='bvls', lsmr_tol='auto', verbose=1)
        #blsqregr=lsq_linear(list(cllayerpt), (df_eg_train[name]['genpart_pt']), bounds = ((meanBounds)/2.0,(meanBounds)*2), method='bvls', lsmr_tol='auto', verbose=1)
        coefflsq=blsqregr.x
        with open('coefflsq_'+pileup+'_'+clustering+'.txt', 'a') as f:
            print(*coefflsq, sep = ", ",end="\n",file=f) 

    if(layercalibration == 'corrPU0'):  ## for the PU200/PU140 case
        with open('coefflsq_PU0_'+clustering+'.txt', 'r') as f:
            for count, line in enumerate(f):
                if(count == name):
                    listcoeff = list(line.split(','))
                    coefflsq = [float(item) for item in listcoeff]

    print("coefflsq", coefflsq)

    ## Corrected Pt (use the corrected layer pT to obtain the total cluster pT)
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
## Now deriving calibration coefficients (using layer corrected cluster pT) using different regression libraries trained on 3D cluster properties
###########

#Defining inputs and targets 

features = ['cl3d_abseta', 'cl3d_n', 'cl3d_pt', 'layer1pt', 'layer2pt', 'layer3pt', 'layer4pt', 'layer5pt', 
  'layer6pt', 'layer7pt', 'layer8pt', 'layer9pt', 'layer10pt', 'layer11pt', 'layer12pt', 'layer13pt', 'layer14pt',
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
# Training calibration 1 using Sklearn LinearRegression
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
## Training calibration 2 gradient boosting techniques

## defining inputs and hyperparameters for GradientBoostingRegressor and XGBoost
input_c2 = {}
target_c2 = {} 

estimator = {}
model_c2_crossVal_huber = {}
model_c2_crossVal_ls = {}
GBR_huber_response_pt_test = {}
GBR_ls_response_pt_test = {}
GBR_feature_importance_huber = [[None for _ in range(len(features))] for _ in range(len(df_eg_train))]
GBR_feature_importance_ls = [[None for _ in range(len(features))] for _ in range(len(df_eg_train))]

model_c3_huber = {}
model_c3_ls = {}
XGB_huber_response_pt_test = {}
XGB_ls_response_pt_test = {}

paramGBR = {}
paramGBR['learning_rate'] = 0.1
paramGBR['n_estimators']  = 200
paramGBR['max_depth'] = 2
paramGBR['subsample'] = 1.0
paramGBR['max_features'] = 1.0
paramGBR['loss']='huber' #huber/ ls / lad

paramXGB_huber = {}
paramXGB_huber['nthread']          	= 10  # limit number of threads
paramXGB_huber['silent'] 			      = True
paramXGB_huber['n_estimators'] 			    = 200  # number of trees to make
paramXGB_huber['objective']   		    = 'reg:pseudohubererror' # objective function
paramXGB_huber['eval_metric']             = 'mphe' ## default for reg:pseudohubererror

paramXGB_ls = {}
paramXGB_ls['nthread']          	= 10  # limit number of threads
paramXGB_ls['silent'] 			      = True
paramXGB_ls['n_estimators'] 			    = 200  # number of trees to make
paramXGB_ls['objective']   		    = 'reg:squarederror' # objective function
paramXGB_ls['eval_metric']             = 'rmse' ## default for reg:squarederror

## Starting BDT calibration from here 

for name in df_eg_train:

    input_c2 = df_eg_train[name][features]
    targetMul_c2 = df_eg_train[name]['targetMul']
    targetAdd_c2 = df_eg_train[name]['targetAdd']
    cl3d_pt_c0 = df_eg_train[name]['cl3d_pt_c0']
    genpart_pt = df_eg_train[name]['genpart_pt']
    cl3d_eta = df_eg_train[name]['cl3d_eta']
    genpart_eta = df_eg_train[name]['genpart_eta']
    cl3d_response_Uncorr = df_eg_train[name]['cl3d_response_Uncorr']
    cl3d_response_c0 = df_eg_train[name]['cl3d_response_c0'] 
    cl3d_response_c1 = df_eg_train[name]['cl3d_response_c1'] 

    print('Training eta calibration with GradientBoostingRegressor')
    print("Starting GBR for FE option: ", name)
    
    ##cross-validation and gridsearch
    X_train, X_test, y_train, y_test, cl3d_pt_train, cl3d_pt_test, genpart_pt_train, genpart_pt_test, cl3d_eta_train, cl3d_eta_test, genpart_eta_train, genpart_eta_test, cl3d_response_Uncorr_train, cl3d_response_Uncorr_test, cl3d_response_c0_train, cl3d_response_c0_test, cl3d_response_c1_train, cl3d_response_c1_test = train_test_split(input_c2, targetMul_c2, cl3d_pt_c0, genpart_pt, cl3d_eta, genpart_eta, cl3d_response_Uncorr, cl3d_response_c0, cl3d_response_c1, test_size=0.4, random_state=0)
    
    ## Use 'tuningGBR' if you want to scan the hyperparameter space for GradientBoostingRegressor
    if(tuningGBR == "True"):
        param_gridsearch = [
            (learning_rate, max_depth, subsample, max_features) 
            for learning_rate in [0.3,0.1,0.07,0.05,0.01]
            for max_depth in [2, 4, 6, 8] 
            for subsample in [0.7,0.8,0.9,1.0]
            for max_features in [0.7,0.8,0.9,1.0]
        ]

        min_maeGBR = float("Inf")
        max_scoreGBR = float(0.0)
        best_paramsGBR = None
        
        ##Looking at the grid of parameters
        for learning_rate, max_depth, subsample, max_features in param_gridsearch:
             print("CV with learning_rate={}, max_depth={}, subsample={}, max_features={}".format(
                   learning_rate,
                   max_depth,
                   subsample, 
                   max_features))
             ## Update our parameters
             paramGBR['learning_rate'] = learning_rate
             paramGBR['max_depth'] = max_depth
             paramGBR['subsample'] = subsample
             paramGBR['max_features'] = max_features
        
             estimator[name] = GradientBoostingRegressor(n_estimators=paramGBR['n_estimators'], learning_rate=paramGBR['learning_rate'], max_depth=paramGBR['max_depth'], max_features = paramGBR['max_features'], subsample = paramGBR['subsample'], random_state=0, loss=paramGBR['loss']).fit(X_train, y_train)
             cv = KFold(n_splits=10, random_state=1, shuffle=True)
             
             ## look at loss funtion for the parameter set
             test_score = np.zeros(200, dtype=np.float64)                                                                     
             for i, y_pred in enumerate(estimator[name].staged_predict(X_test)):
                 test_score[i] = estimator[name].loss_(y_test, y_pred)      
             fig = plt.figure(figsize=(6, 6))
             plt.subplot(1, 1, 1)
             plt.title('Deviance '+fe_names[name])
             plt.plot(np.arange(200), estimator[name].train_score_, 'b-', label='Training Set Deviance')
             plt.plot(np.arange(200), test_score, 'r-', label='Test Set Deviance')
             plt.legend(loc='upper right')
             plt.xlabel('Boosting Iterations')
             plt.ylabel('Deviance')
             fig.tight_layout()
             plt.savefig(plotdir+'GBR_training_deviance_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(subsample)+'_'+str(max_features)+'_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')         
             plt.savefig(plotdir+'GBR_training_deviance_'+str(learning_rate)+'_'+str(max_depth)+'_'+str(subsample)+'_'+str(max_features)+'_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')         

             # Update best MAE
             mean_m = np.mean(-cross_val_score(estimator[name], X=X_train, y=y_train, scoring = "neg_mean_absolute_error", cv = cv))
             mean_mse = np.mean(-cross_val_score(estimator[name], X=X_train, y=y_train, scoring = "neg_mean_squared_error", cv = cv))
             mean_score = np.mean(cross_val_score(estimator[name], X=X_train, y=y_train, scoring = "r2", cv = cv))
             print("\tMAE {}, MSE {}, r2 {}".format(mean_m, mean_mse, mean_score))
        #    if(mean_m < min_maeGBR):
             if(mean_score > max_scoreGBR):
                 min_maeGBR = mean_m
                 min_mseGBR = mean_mse
                 max_scoreGBR = mean_score
                 best_paramsGBR = (learning_rate,max_depth,subsample,max_features)
        print("Best params: {}, {}, {}, {}, MAE: {}, MSE: {}, MaxScore: {}".format(best_paramsGBR[0], best_paramsGBR[1], best_paramsGBR[2], best_paramsGBR[3], min_maeGBR, min_mseGBR, max_scoreGBR))

    ## Defining our final GBR models (use values obtained from tuning above)
    if(name == 0):
        model_c2_crossVal_huber[name] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, subsample = 1.0, max_features = 1.0, random_state=0, loss='huber').fit(X_train, y_train)
        model_c2_crossVal_ls[name] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, subsample = 1.0, max_features = 1.0, random_state=0, loss='ls').fit(X_train, y_train)
    else:
        model_c2_crossVal_huber[name] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2, subsample = 1.0, max_features = 1.0, random_state=0, loss='huber').fit(X_train, y_train)
        model_c2_crossVal_ls[name] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2, subsample = 1.0, max_features = 1.0, random_state=0, loss='ls').fit(X_train, y_train)

    print("huber GBR train score: ",model_c2_crossVal_huber[name].score(X_train, y_train))
    print("huber GBR test score: ",model_c2_crossVal_huber[name].score(X_test, y_test))
    print("huber GBR MSE: ",mean_squared_error(y_test, model_c2_crossVal_huber[name].predict(X_test)))
    print("ls GBR train score: ",model_c2_crossVal_ls[name].score(X_train, y_train))
    print("ls GBR test score: ",model_c2_crossVal_ls[name].score(X_test, y_test))
    print("ls GBR MSE: ",mean_squared_error(y_test, model_c2_crossVal_ls[name].predict(X_test)))

    ## plotting training deviance for the chosen models
    test_score200 = np.zeros(200, dtype=np.float64)                                                                           
    test_score100 = np.zeros(200, dtype=np.float64)                                                                           
    print("ploting deviance huber")
    if(name == 0):
        for i, y_pred in enumerate(model_c2_crossVal_huber[name].staged_predict(X_test)):
            test_score200[i] = model_c2_crossVal_huber[name].loss_(y_test, y_pred)      
    else:
        for i, y_pred in enumerate(model_c2_crossVal_huber[name].staged_predict(X_test)):
            test_score100[i] = model_c2_crossVal_huber[name].loss_(y_test, y_pred)      
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title('Deviance '+fe_names[name])
    if(name == 0):
        plt.plot(np.arange(200), model_c2_crossVal_huber[name].train_score_, 'b-', label='Training Set Deviance')
        plt.plot(np.arange(200), test_score200, 'r-', label='Test Set Deviance')
    else:
        plt.plot(np.arange(200), model_c2_crossVal_huber[name].train_score_, 'b-', label='Training Set Deviance')
        plt.plot(np.arange(200), test_score100, 'r-', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.savefig(plotdir+'GBR_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')         
    plt.savefig(plotdir+'GBR_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')         
   
    print("ploting deviance ls")
    test_scorelsa = np.zeros(200, dtype=np.float64)
    test_scorelsb = np.zeros(200, dtype=np.float64)

    if(name == 0):                                                                           
        for i, y_pred in enumerate(model_c2_crossVal_ls[name].staged_predict(X_test)):
            test_scorelsa[i] = model_c2_crossVal_ls[name].loss_(y_test, y_pred)      
    else:
        for i, y_pred in enumerate(model_c2_crossVal_ls[name].staged_predict(X_test)):
            test_scorelsb[i] = model_c2_crossVal_ls[name].loss_(y_test, y_pred)      
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title('Deviance '+fe_names[name])
    if(name == 0):
        plt.plot(np.arange(200), model_c2_crossVal_ls[name].train_score_, 'b-', label='Training Set Deviance')
        plt.plot(np.arange(200), test_scorelsa, 'r-', label='Test Set Deviance')
    else:
        plt.plot(np.arange(200), model_c2_crossVal_ls[name].train_score_, 'b-', label='Training Set Deviance')
        plt.plot(np.arange(200), test_scorelsb, 'r-', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.savefig(plotdir+'GBR_training_deviance_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')         
    plt.savefig(plotdir+'GBR_training_deviance_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')         

    ## Results on test sample GBR
    GBR_huber_predict_test = model_c2_crossVal_huber[name].predict(X_test)
    GBR_huber_corr_pt_test = (cl3d_pt_test) * (GBR_huber_predict_test)
    GBR_huber_response_pt_test[name] = GBR_huber_corr_pt_test / genpart_pt_test
    GBR_feature_importance_huber[name] = model_c2_crossVal_huber[name].feature_importances_

    GBR_ls_predict_test = model_c2_crossVal_ls[name].predict(X_test)
    GBR_ls_corr_pt_test = (cl3d_pt_test) * (GBR_ls_predict_test)
    GBR_ls_response_pt_test[name] = GBR_ls_corr_pt_test / genpart_pt_test
    GBR_feature_importance_ls[name] = model_c2_crossVal_ls[name].feature_importances_

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

    print('Training eta calibration with xgboost')
    print("Starting xgboost for FE option: ", name)

    train = xgb.DMatrix(data=X_train,label=y_train, feature_names=features) 
    test = xgb.DMatrix(data=X_test,label=y_test,feature_names=features)
 
    ## Use 'tuningXGBOOST' if you want to scan the hyperparameter space for XGBOOST
    if(tuningXGBOOST == 'True'):
        params = {
            # Parameters that we are going to tune.                                                                           
            'max_depth':2,
            'eta':.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            # Other parameters                                                                                                 
            'objective': 'reg:pseudohubererror' #'reg:pseudohubererror'  #'objective':'reg:squarederror'
        }

        gridsearch_params = [
            (eta, max_depth, subsample, colsample_bytree)
            for eta in [0.3,0.1,0.07,0.05,0.01]
            for max_depth in [2,4,6,8]
            for subsample in [i/10. for i in range(7,11)]
            for colsample_bytree in [i/10. for i in range(7,11)]
        ]

        min_mae = float("Inf")
        best_params = None

        for eta, max_depth, subsample, colsample_bytree in gridsearch_params:
            print("CV with eta={}, max_depth={}, subsample={}, colsample_bytree={}".format(
                eta,
                max_depth,
                subsample, 
                colsample_bytree))
            # Update our parameters
            params['eta'] = eta
            params['max_depth'] = max_depth
            params['subsample'] = subsample
            params['colsample_bytree'] = colsample_bytree
            
            ## looking at loss 
            ntrees = paramXGB_huber['n_estimators']
            progress = dict()
            watchlist = [(test, 'eval'), (train, 'train')]
            booster = xgb.train(params, train, ntrees, watchlist, evals_result=progress) 
            ## retrieve performance metrics
            epochs = len(progress['train']['mphe']) ## mphe, rmse
            x_axis = range(0, epochs)
            # plot loss function
            fig, ax = plt.subplots(figsize=(12,12))
            ax.plot(x_axis, progress['train']['mphe'], label='Train')
            ax.plot(x_axis, progress['eval']['mphe'], label='Test')
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.legend(frameon=False,fontsize=20)
            plt.xlabel('Boosting Iterations',fontsize=20)
            plt.ylabel('Error',fontsize=20)
            plt.title('XGBoost Loss , '+fe_names[name])
            plt.savefig(plotdir+'xgboost_training_deviance_'+str(eta)+'_'+str(max_depth)+'_'+str(subsample)+'_'+str(colsample_bytree)+'_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')         
            plt.savefig(plotdir+'xgboost_training_deviance_'+str(eta)+'_'+str(max_depth)+'_'+str(subsample)+'_'+str(colsample_bytree)+'_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')         

            cv_results = xgb.cv(
                params,
                train,
                num_boost_round=paramXGB_huber['n_estimators'],
                seed=42,
                nfold=5,
                metrics={'mae'}, ##mphe, mae
                early_stopping_rounds=10
            )

            # Update best MAE
            mean_mphe = cv_results['test-mae-mean'].min() ## test-mphe-mean, test-mae-mean
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMPHE {} for {} rounds".format(mean_mphe, boost_rounds))
            if mean_mphe < min_mae:
                min_mae = mean_mphe
                best_params = (eta,max_depth,subsample,colsample_bytree)
        print("Best params: {}, {}, {}, {}, MAE: {}".format(best_params[0], best_params[1], best_params[2], best_params[3], min_mae))


    #### Defining the final models (values used from tuning)
    if(name == 0):
        paramXGB_huber['eta']              	= 0.07 # learning rate
        paramXGB_huber['max_depth']        	= 2  # maximum depth of a tree
        paramXGB_huber['subsample']        	= 1.0 # fraction of events to train tree on
        paramXGB_huber['colsample_bytree'] 	= 1.0 # fraction of features to train tree on
        paramXGB_huber['n_estimators'] = 200

        paramXGB_ls['eta']              	= 0.05 # learning rate
        paramXGB_ls['max_depth']        	= 2  # maximum depth of a tree
        paramXGB_ls['subsample']        	= 1.0 # fraction of events to train tree on
        paramXGB_ls['colsample_bytree'] 	= 1.0 # fraction of features to train tree on
        paramXGB_ls['n_estimators'] = 200
    else:
        paramXGB_huber['eta']              	= 0.1 # learning rate
        paramXGB_huber['max_depth']        	= 2  # maximum depth of a tree
        paramXGB_huber['subsample']        	= 1.0 # fraction of events to train tree on
        paramXGB_huber['colsample_bytree'] 	= 1.0 # fraction of features to train tree on
        paramXGB_huber['n_estimators'] = 200

        paramXGB_ls['eta']              	= 0.1 # learning rate
        paramXGB_ls['max_depth']        	= 2  # maximum depth of a tree
        paramXGB_ls['subsample']        	= 1.0 # fraction of events to train tree on
        paramXGB_ls['colsample_bytree'] 	= 1.0 # fraction of features to train tree on
        paramXGB_ls['n_estimators'] = 200

    print("Hyperparameters xgboost huber: ", paramXGB_huber)
    print("Hyperparameters xgboost ls: ", paramXGB_ls)

    ## plot the training deviance for final XGBOOST models 
    output = 'targetMul' 
    progress = dict()
    watchlist = [(test, 'eval'), (train, 'train')]
    ntrees = paramXGB_huber['n_estimators']
    model_c3_huber[name] = xgb.train(paramXGB_huber, train, ntrees, watchlist, evals_result=progress)
    ## retrieve performance metrics
    epochs = len(progress['train']['mphe']) ## mphe, rmse
    x_axis = range(0, epochs)
    # plot loss function
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, progress['train']['mphe'], label='Train')
    ax.plot(x_axis, progress['eval']['mphe'], label='Test')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(frameon=False,fontsize=20)
    plt.xlabel('Boosting Iterations',fontsize=20)
    plt.ylabel('Error',fontsize=20)
    plt.title('XGBoost Loss , '+fe_names[name])
    plt.savefig(plotdir+'xgboost_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
    plt.savefig(plotdir+'xgboost_training_deviance_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')
    print("xgb huber model done")

    ntrees = paramXGB_ls['n_estimators']
    model_c3_ls[name] = xgb.train(paramXGB_ls, train, ntrees, watchlist, evals_result=progress) 
    ## retrieve performance metrics
    epochs = len(progress['train']['rmse']) ## mphe, rmse
    x_axis = range(0, epochs)
    # plot loss function
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, progress['train']['rmse'], label='Train')
    ax.plot(x_axis, progress['eval']['rmse'], label='Test')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(frameon=False,fontsize=20)
    plt.xlabel('Boosting Iterations',fontsize=20)
    plt.ylabel('Error',fontsize=20)
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Error')
    plt.title('XGBoost Loss , '+fe_names[name])
    plt.savefig(plotdir+'xgboost_training_deviance_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
    plt.savefig(plotdir+'xgboost_training_deviance_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')
    print("xgb ls model done")

    ## Results on test sample XGBOOST
    XGB_huber_predict_test = model_c3_huber[name].predict(test)
    XGB_huber_corr_pt_test = (cl3d_pt_test) * (XGB_huber_predict_test)
    XGB_huber_response_pt_test[name] = XGB_huber_corr_pt_test / genpart_pt_test

    XGB_ls_predict_test = model_c3_ls[name].predict(test)
    XGB_ls_corr_pt_test = (cl3d_pt_test) * (XGB_ls_predict_test)
    XGB_ls_response_pt_test[name] = XGB_ls_corr_pt_test / genpart_pt_test

    plt.figure(figsize=(15,10))
    xgb.plot_importance(model_c3_huber[name], grid=False, importance_type='gain',lw=2)
    plt.title('XGBOOST Regressor')
    plt.subplots_adjust(left=0.50, right=0.85, top=0.9, bottom=0.2)
    plt.savefig(plotdir+'XGBOOST_bdt_importances_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
    plt.savefig(plotdir+'XGBOOST_bdt_importances_huber_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

    plt.figure(figsize=(15,10))
    xgb.plot_importance(model_c3_ls[name], grid=False, importance_type='gain',lw=2)
    plt.title('XGBOOST Regressor')
    plt.subplots_adjust(left=0.50, right=0.85, top=0.9, bottom=0.2)
    plt.savefig(plotdir+'XGBOOST_bdt_importances_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
    plt.savefig(plotdir+'XGBOOST_bdt_importances_ls_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

    ########
    ## Plotting the final results from LinearRegression, GradientBoostingRegressor and XGBOOST 
    df_test = pd.DataFrame(genpart_eta_test)
    df_test['genpart_abseta'] = np.abs(genpart_eta_test)
    df_test['genpart_pt'] = genpart_pt_test
    df_test['cl3d_abseta'] = np.abs(cl3d_eta_test)
    df_test['cl3d_pt'] = cl3d_pt_test
    df_test['cl3d_response_Uncorr'] = cl3d_response_Uncorr_test
    df_test['cl3d_response_c0'] = cl3d_response_c0_test 
    df_test['cl3d_response_c1'] = cl3d_response_c1_test 
    df_test['cl3d_response_c2'] = GBR_huber_response_pt_test[name]
    df_test['cl3d_response_c2b'] = GBR_ls_response_pt_test[name]
    df_test['cl3d_response_c3'] = XGB_huber_response_pt_test[name]
    df_test['cl3d_response_c3b'] = XGB_ls_response_pt_test[name]
    df_test['bineta'] = ((df_test['genpart_abseta'] - 1.6)/0.13).astype('int32')
    df_test['binpt'] = ((df_test['genpart_pt']- 10.0)/10.0).astype('int32')
    df_mean_eta = df_test.groupby(['bineta']).mean()
    df_mean_pt = df_test.groupby(['binpt']).mean()
    df_effrms_eta = df_test.groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_Uncorr))
    df_effrms_pt = df_test.groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_Uncorr))
    df_effrms_etaC1 = df_test.groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c1))
    df_effrms_ptC1 = df_test.groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c1)) 
    df_effrms_etaC2 = df_test.groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c2))
    df_effrms_ptC2 = df_test.groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c2)) 
    df_effrms_etaC2b = df_test.groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c2b))
    df_effrms_ptC2b = df_test.groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c2b)) 
    df_effrms_etaC3 = df_test.groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c3))
    df_effrms_ptC3 = df_test.groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c3)) 
    df_effrms_etaC3b = df_test.groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c3b))
    df_effrms_ptC3b = df_test.groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c3b)) 

    print("plotting start")

    if(name==0):
        fig = plt.figure(num='performance',figsize=(32,48))
        plt.title(pileup+'_'+clustering)
    plt.figure(num='performance')
    plt.subplot(641)
    plt.title('Uncorrected')
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(642)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(643) 
    plt.errorbar((df_mean_eta.genpart_abseta), df_effrms_eta/df_mean_eta.cl3d_response_Uncorr, linestyle='-', marker='o',  label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(644) 
    plt.errorbar((df_mean_pt.genpart_pt), df_effrms_pt/df_mean_pt.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(645)
    plt.title('Linear Regression')
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(646)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(647)
    plt.errorbar((df_mean_eta.genpart_abseta),df_effrms_etaC1/df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(648)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC1/df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(649)
    plt.title('Gradient Boosting Regressor (loss=huber)')
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,10)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,11)
    plt.errorbar((df_mean_eta.genpart_abseta),df_effrms_etaC2/df_mean_eta.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,12)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC2/df_mean_pt.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(6,4,13)
    plt.title('Gradient Boosting Regressor (loss=ls)')
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,14)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,15)
    plt.errorbar((df_mean_eta.genpart_abseta),df_effrms_etaC2b/df_mean_eta.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,16)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC2b/df_mean_pt.cl3d_response_c2b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(6,4,17)
    plt.title('XGBOOST Regression (huber)')
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,18)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,19)
    plt.errorbar((df_mean_eta.genpart_abseta),df_effrms_etaC3/df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,20)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC3/df_mean_pt.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(6,4,21)
    plt.title('XGBOOST Regression (ls)')
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_c3b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,22)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c3b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.85,1.2)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,23)
    plt.errorbar((df_mean_eta.genpart_abseta),df_effrms_etaC3b/df_mean_eta.cl3d_response_c3b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(6,4,24)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC3b/df_mean_pt.cl3d_response_c3b, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.15)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
#    plt.subplots_adjust(left=0.28, right=0.85, top=0.9, bottom=0.1)
    plt.savefig(plotdir+'calibration_performancesummary_'+pileup+'_'+clustering+'.png')
    plt.savefig(plotdir+'calibration_performancesummary_'+pileup+'_'+clustering+'.pdf')

    if(name==0):
        fig = plt.figure(num='performanceBest',figsize=(32,10))
        plt.title(pileup+'_'+clustering)
    plt.figure(num='performanceBest')
    plt.subplot(141)
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name]+'LR')
    plt.errorbar((df_mean_eta.genpart_abseta), df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name]+'Xgboost (huber)')
    plt.ylim(0.85,1.2)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(142)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name]+'LR')
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name]+'Xgboost (huber)')
    plt.ylim(0.85,1.2)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(143)
    plt.errorbar((df_mean_eta.genpart_abseta),df_effrms_etaC1/df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name]+'LR') 
    plt.errorbar((df_mean_eta.genpart_abseta),df_effrms_etaC3/df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name]+'Xgboost (huber)')
    plt.ylim(0.01,0.15)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(144)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC1/df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name]+'LR')
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC3/df_mean_pt.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name]+'Xgboost (huber)')
    plt.ylim(0.01,0.15)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.savefig(plotdir+'calibration_performanceLR_BDTbest_'+pileup+'_'+clustering+'.png')
    plt.savefig(plotdir+'calibration_performanceLR_BDTbest_'+pileup+'_'+clustering+'.pdf')

    plt.figure(figsize=(15,10))
    plt.title(fe_names[name])
    plt.hist((df_test.cl3d_response_Uncorr), bins=np.arange(0.0, 1.4, 0.01), label = 'Uncorrected', histtype = 'step')
    plt.hist((df_test.cl3d_response_c0), bins=np.arange(0.0, 1.4, 0.01), label = 'Layer weights (C)', histtype = 'step')
    plt.hist((df_test.cl3d_response_c1), bins=np.arange(0.0, 1.4, 0.01), label = 'C + LR', histtype = 'step')
    plt.hist((df_test.cl3d_response_c2), bins=np.arange(0.0, 1.4, 0.01), label = 'C + GBR (huber)', histtype = 'step')
    plt.hist((df_test.cl3d_response_c2b), bins=np.arange(0.0, 1.4, 0.01), label = 'C + GBR (ls)', histtype = 'step')
    plt.hist((df_test.cl3d_response_c3),  bins=np.arange(0.0, 1.4, 0.01),  label = 'C + xgboost (huber)', histtype = 'step')
    plt.hist((df_test.cl3d_response_c3b),  bins=np.arange(0.0, 1.4, 0.01),  label = 'C + xgboost (ls)', histtype = 'step')
    plt.xlabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.xlim(0.2,1.4)
    plt.legend(frameon=False)    
    plt.grid(False)
    plt.subplots_adjust(left=0.28, right=0.85, top=0.9, bottom=0.1)
    plt.savefig(plotdir+'calibration_response_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
    plt.savefig(plotdir+'calibration_response_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

    print("plotting done")

print("calibration done")

#for name in df_eg_train:

#    with open(file_out_model_c2[name], 'wb') as f:
#        pickle.dump(model_c2_crossVal_huber[name], f)

###########

# Results

###########

# Save files

#for name in df_eg:

#  store_eg = pd.HDFStore(file_out_eg[name], mode='w')
#  store_eg['df_eg_PU200'] = df_eg_train[name]
#  store_eg.close()

###########


