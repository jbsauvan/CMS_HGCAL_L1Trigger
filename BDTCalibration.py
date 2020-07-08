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
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy.optimize import lsq_linear
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
#import root_pandas
###########

def train_xgb(df, inputs, output, hyperparams, test_fraction=0.4):
    X_train, X_test, y_train, y_test = train_test_split(df[inputs], df[output], test_size=test_fraction)#random_state=123)
    train = xgb.DMatrix(data=X_train,label=y_train, feature_names=inputs) 
    test = xgb.DMatrix(data=X_test,label=y_test,feature_names=inputs) 
    full = xgb.DMatrix(data=df[inputs],label=df[output],feature_names=inputs) 
    booster = xgb.train(hyperparams, full)#, num_boost_round=hyperparams['num_trees']) 
    df['bdt_output'] = booster.predict(full)

    return booster, df['bdt_output']

def efficiency(group, cut):
    tot = group.shape[0]
    sel = group[group.cl3d_pubdt_score > cut].shape[0]
    return float(sel)/float(tot)

###########

fe_names = {}

#fe_names[0] = 'Thr'  
#fe_names[0] = 'BC2'#'STC'
#fe_names[1] = 'BC4'#'BC'
#fe_names[2] = 'BC8'#'BCCoarse'  
fe_names[0] = 'BC_STC'

algo_name=['F8Tdr015','F8BC2','F8BC4','F8BC8','F8BCSTC']

###########

dir_in = '/data_CMS/cms/ahuja/HGCAL/ntuples/tpgV3150/outhdffiles/RelValV10/RelValDiG_Pt10To100_Eta1p6To2p9_xydr015PU0/'

file_in_eg = {}

#file_in_eg[0] = dir_in+'F8Tdr015.hdf5'
#file_in_eg[0] = dir_in+'F8BC2.hdf5'
#file_in_eg[1] = dir_in+'F8BC4.hdf5'
#file_in_eg[2] = dir_in+'F8BC8.hdf5'
file_in_eg[0] = dir_in+'F8BCSTC.hdf5'

###########

dir_out = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/'

file_out_eg = {}

#file_out_eg[0] = dir_out+'ntuple_THRESHOLD_pubdt_algoA.hdf5'
#file_out_eg[0] = dir_out+'ntuple_BESTCHOICE2_bdt.hdf5' #SUPERTRIGGERCELL
#file_out_eg[1] = dir_out+'ntuple_BESTCHOICE4_bdt.hdf5'
#file_out_eg[2] = dir_out+'ntuple_BESTCHOICECOARSE_bdt.hdf5'
file_out_eg[0] = dir_out+'ntuple_MIXEDBCSTC_bdt.hdf5'

###########

dir_out_model = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/models/'

file_out_model_c1 = {}

#file_out_model[0] = dir_out_model+'model_LR_threshold.pkl'
#file_out_model_c1[0] = dir_out_model+'model_LR_bestchoice2.pkl'
#file_out_model_c1[1] = dir_out_model+'model_LR_bestchoice4.pkl'
#file_out_model_c1[2] = dir_out_model+'model_LR_bestchoicecoarse.pkl'
file_out_model_c1[0] = dir_out_model+'model_LR_mixedbcstc.pkl'

file_out_model_c2 = {}

#file_out_model_c2[0] = dir_out_model+'model_GBR_threshold.pkl'
#file_out_model_c2[0] = dir_out_model+'model_GBR_bestchoice2.pkl' ##supertriggercell
#file_out_model_c2[1] = dir_out_model+'model_GBR_bestchoice4.pkl'
#file_out_model_c2[2] = dir_out_model+'model_GBR_bestchoicecoarse.pkl'
file_out_model_c2[0] = dir_out_model+'model_GBR_mixedbcstc.pkl'

file_out_model_c3 = {}

#file_out_model_c3[0] = dir_out_model+'model_xgboost_threshold.pkl'
#file_out_model_c3[0] = dir_out_model+'model_xgboost_bestchoice2.pkl'
#file_out_model_c3[1] = dir_out_model+'model_xgboost_bestchoice4.pkl'
#file_out_model_c3[2] = dir_out_model+'model_xgboost_bestchoicecoarse.pkl'
file_out_model_c3[0] = dir_out_model+'model_xgboost_mixedbcstc.pkl'

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

  sel = df_eg_train[name]['cl3d_pt'] > 4
  df_eg_train[name] = df_eg_train[name][sel]

  events_stored_eg[name] = np.unique(df_eg_train[name].reset_index()['event']).shape[0]

print(' ')

###########

## Correcting layer pt with bounded least square
# Training calibration 0 
print('Training calibration for layer pT with lsq_linear')

model_c1 = {}
calibration = 'direct'

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
    #print(df_eg_train[name]['cl3d_layerptsum']/df_eg_train[name]['cl3d_pt'])

    ##uncorrected resolution
    mean_Unc_reso = np.mean((df_eg_train[name]['cl3d_pt'])/(df_eg_train[name]['genpart_pt']))
    meanBounds = 1/mean_Unc_reso
    ##coefficients
    if(calibration == 'direct'):
        blsqregr=lsq_linear(list(cllayerpt), (df_eg_train[name]['genpart_pt']), bounds = (0.5,2.0), method='bvls', lsmr_tol='auto', verbose=1)
        #blsqregr=lsq_linear(list(cllayerpt), (df_eg_train[name]['genpart_pt']), bounds = ((meanBounds)/2.0,(meanBounds)*2), method='bvls', lsmr_tol='auto', verbose=1)
    coefflsq=blsqregr.x

    print(*coefflsq, sep = ", ")
    with open('coefflsq.txt', 'a') as f:
        print("[",end="",file=f)
        print(*coefflsq, sep = ", ",end="],\n",file=f) 

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

#Defining target 

for name in df_eg_train:

  df_eg_train[name]['cl3d_PU'] = np.abs(df_eg_train[name].genpart_pt - df_eg_train[name].cl3d_pt_c0)
  print(df_eg_train[name]['cl3d_PU'])

###########

# Training calibration 1 
print('Training eta calibration with LinearRegression')

for name in df_eg_train:

  input_c1 = df_eg_train[name][['cl3d_abseta']] 
  target_c1 = df_eg_train[name]['cl3d_PU']
  model_c1[name] = LinearRegression().fit(input_c1, target_c1)

for name in df_eg_train:

  with open(file_out_model_c1[name], 'wb') as f:
    pickle.dump(model_c1[name], f)

for name in df_eg_train:

  df_eg_train[name]['cl3d_c1'] = model_c1[name].predict(df_eg_train[name][['cl3d_abseta']]) 
  print("LR coeff:", df_eg_train[name]['cl3d_c1'])
  df_eg_train[name]['cl3d_pt_c1'] = df_eg_train[name].cl3d_pt - ((df_eg_train[name].cl3d_c1)*(df_eg_train[name].cl3d_PU))
  #df_eg_train[name]['cl3d_pt_c1'] = df_eg_train[name]['cl3d_pt_c0']-(((model_c1[name].coef_)*df_eg_train[name][['cl3d_abseta']])+(model_c1[name].intercept_))
  df_eg_train[name]['cl3d_response_c1'] = df_eg_train[name].cl3d_pt_c1 / df_eg_train[name].genpart_pt

###########

# Training calibration 2
print('Training eta calibration with GradientBoostingRegressor')

#features = ['n_matched_cl3d', 'cl3d_abseta', 
#  'cl3d_showerlength', 'cl3d_coreshowerlength', 
#  'cl3d_firstlayer', 'cl3d_maxlayer', 
#  'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean',
#  'cl3d_hoe', 'cl3d_meanz', 
#  'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 
#  'cl3d_ntc67', 'cl3d_ntc90']

features = ['cl3d_abseta','cl3d_n', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer',
       'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz',
       'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_emaxe']

model_c2 = {}

for name in df_eg_train:

  input_c2 = df_eg_train[name][features]
  target_c2 = df_eg_train[name]['cl3d_PU']
  model_c2[name] = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=2, random_state=0, loss='huber').fit(input_c2, target_c2)

for name in df_eg_train:

  with open(file_out_model_c2[name], 'wb') as f:
    pickle.dump(model_c2[name], f)

for name in df_eg_train:

  df_eg_train[name]['cl3d_c2'] = model_c2[name].predict(df_eg_train[name][features])
  print("GBR coeff:",df_eg_train[name]['cl3d_c2'])
  df_eg_train[name]['cl3d_pt_c2'] = df_eg_train[name].cl3d_pt_c0 - ((df_eg_train[name].cl3d_c2)*(df_eg_train[name].cl3d_PU))
  df_eg_train[name]['cl3d_response_c2'] = df_eg_train[name].cl3d_pt_c2 / df_eg_train[name].genpart_pt
  print(df_eg_train[name]['cl3d_response_c2'])
###########

# Training calibration 2 (xgboost)
print('Training eta calibration with xgboost')

#inputs = ['cl3d_abseta','cl3d_showerlength','cl3d_coreshowerlength',
#   'cl3d_firstlayer', 'cl3d_maxlayer',
#   'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean',
#   'cl3d_hoe', 'cl3d_meanz',
#   'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90',
#   'cl3d_ntc67', 'cl3d_ntc90']

inputs = ['cl3d_abseta','cl3d_n', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_maxlayer',
       'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz',
       'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_emaxe']

param = {}

param['nthread']          	= 10  # limit number of threads
param['eta']              	= 0.2 # learning rate
param['max_depth']        	= 4  # maximum depth of a tree
param['subsample']        	= 0.8 # fraction of events to train tree on
param['colsample_bytree'] 	= 0.8 # fraction of features to train tree on
param['silent'] 			      = True
param['objective']   		    = 'reg:squarederror' #'reg:pseudohubererror' # objective function
#param['num_trees'] 			    = 162  # number of trees to make
#param['eval_metric']             = 'mphe' ## default for reg:pseudohubererror

model_c3 = {}

for name in df_eg_train:

  output = 'cl3d_PU' 
  model_c3[name], df_eg_train[name]['output_c3']= train_xgb(df_eg_train[name], inputs, output, param, test_fraction=0.4)
  ##cv_results = xgb.cv(param)
  ##print(cv_results)
    
for name in df_eg_train:

  with open(file_out_model_c3[name], 'wb') as f:
    pickle.dump(model_c3[name], f)

for name in df_eg_train:

  full = xgb.DMatrix(data=df_eg_train[name][inputs], label=df_eg_train[name][output], feature_names=inputs)
  df_eg_train[name]['cl3d_c3'] = model_c3[name].predict(full)
  print("model 3 coeff", df_eg_train[name]['cl3d_c3'])
  df_eg_train[name]['cl3d_pt_c3'] = df_eg_train[name].cl3d_pt_c0 - ((df_eg_train[name].cl3d_c3)*(df_eg_train[name].cl3d_PU))
  df_eg_train[name]['cl3d_response_c3'] = df_eg_train[name].cl3d_pt_c3 / df_eg_train[name].genpart_pt

print(' ')

###########

# Application

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
#legends[0] = 'Threshold 1.35 mipT'
legends[0] = 'BC2 '#'STC4+16'
legends[1] = 'BC4' #'BC Decentral'
legends[2] = 'BC Coarse 2x2 TC'
legends[3] = 'Mixed BC + STC'

#matplotlib.rcParams.update({'font.size': 22})
#plt.figure(figsize=(15,10))

# FEATURE IMPORTANCES

matplotlib.rcParams.update({'font.size': 16})

for name in df_eg_train:

  plt.figure(figsize=(15,10))
  xgb.plot_importance(model_c3[name], grid=False, importance_type='gain',lw=2)
  plt.subplots_adjust(left=0.50, right=0.85, top=0.9, bottom=0.2)
  plt.savefig(plotdir+'bdt_importances_'+fe_names[name]+'.png')
  plt.savefig(plotdir+'bdt_importances_'+fe_names[name]+'.pdf')

#for name in df_eg_train:

#  plt.figure(figsize=(12,10))
#  plt.hist(df_eg[name]['cl3d_bdt_score'], bins=np.arange(-0.2, 1.2, 0.02), normed=True, color='red', histtype='step', lw=2, label='Eg PU=200')
#  plt.legend(loc = 'upper right', fontsize=22)
#  plt.xlabel(r'PU BDT score')
#  plt.ylabel(r'Entries')
#  plt.savefig(plotdir+'bdt_scores_'+fe_names[name]+'.png')
#  plt.savefig(plotdir+'bdt_scores_'+fe_names[name]+'.pdf')

# Calibration improvement 

for name in df_eg_train:

  plt.figure(figsize=(15,10))
  plt.errorbar(df_eg_train[name].cl3d_pt, np.abs(df_eg_train[name].cl3d_response_c0),  label = 'Corrected Layer $p_{T}$ (C)')
  plt.errorbar(df_eg_train[name].cl3d_pt, np.abs(df_eg_train[name].cl3d_response_c1),  label = 'C + LR')
  plt.errorbar(df_eg_train[name].cl3d_pt, np.abs(df_eg_train[name].cl3d_response_c2),  label = 'C + GBR')
  plt.errorbar(df_eg_train[name].cl3d_pt, np.abs(df_eg_train[name].cl3d_response_c3),  label = 'C + xgboost')
  plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
  plt.ylabel('Calibrated response',fontsize=20)
  plt.legend(frameon=False)    
  plt.grid(False)
  plt.subplots_adjust(left=0.28, right=0.85, top=0.9, bottom=0.1)
  plt.savefig(plotdir+'calibration_response_'+fe_names[name]+'.png')
  plt.savefig(plotdir+'calibration_response_'+fe_names[name]+'.pdf')

###########

