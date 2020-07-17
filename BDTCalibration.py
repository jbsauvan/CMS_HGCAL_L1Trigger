# S. Ahuja, sudha.ahuja@cern.ch, June 202<0

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
fe_names[1] = 'Best Choice'
fe_names[2] = 'STC'
fe_names[3] = 'BC+STC'
fe_names[4] = 'BC course4'

pileup = 'PU200' ## PU0, PU140, PU200
clustering = 'drdefault' ## drdefault, dr015, drOptimal
layercalibration = 'corrPU0' ## derive, corrPU0 

###########

dir_in1 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_th/'
dir_in2 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_bcdcen_th/'
dir_in3 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_ctc/'
dir_in4 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_mixedfe/'

file_in_eg = {}

file_in_eg[0] = dir_in1+'Thr.hdf5'
file_in_eg[1] = dir_in2+'BCdcen.hdf5'
file_in_eg[2] = dir_in3+'STC.hdf5'
file_in_eg[3] = dir_in4+'BCSTC.hdf5'
file_in_eg[4] = dir_in4+'BCcourse4.hdf5'

###########

dir_out = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/'

file_out_eg = {}

file_out_eg[0] = dir_out+'ntuple_'+pileup+'_'+clustering+'_THRESHOLD_bdt.hdf5'
file_out_eg[1] = dir_out+'ntuple_'+pileup+'_'+clustering+'_BESTCHOICEDCEN_bdt.hdf5'
file_out_eg[2] = dir_out+'ntuple_'+pileup+'_'+clustering+'_SUPERTRIGGERCELL_bdt.hdf5'
file_out_eg[3] = dir_out+'ntuple_'+pileup+'_'+clustering+'_MIXEDBCSTC_bdt.hdf5'
file_out_eg[4] = dir_out+'ntuple_'+pileup+'_'+clustering+'_BESTCHOICECOURSE4_bdt.hdf5'

###########

dir_out_model = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/models/'

file_out_model_c1 = {}

file_out_model_c1[0] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_threshold.pkl'
file_out_model_c1[1] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_bestchoicedcen.pkl'
file_out_model_c1[2] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_supertriggercell.pkl'
file_out_model_c1[3] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_mixedbcstc.pkl'
file_out_model_c1[4] = dir_out_model+'model_LR_'+pileup+'_'+clustering+'_bestchoicecourse4.pkl'

file_out_model_c2 = {}

file_out_model_c2[0] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_threshold.pkl'
file_out_model_c2[1] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_bestchoicedcen.pkl'
file_out_model_c2[2] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_supertriggercell.pkl'
file_out_model_c2[3] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_mixedbcstc.pkl'
file_out_model_c2[4] = dir_out_model+'model_GBR_'+pileup+'_'+clustering+'_bestchoicecourse4.pkl'

file_out_model_c3 = {}

file_out_model_c3[0] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_threshold.pkl'
file_out_model_c3[1] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_bestchoicedcen.pkl'
file_out_model_c3[2] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_supertriggercell.pkl'
file_out_model_c3[3] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_mixedbcstc.pkl'
file_out_model_c3[4] = dir_out_model+'model_xgboost_'+pileup+'_'+clustering+'_bestchoicecourse4.pkl'

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

#Defining target 

for name in df_eg_train:

  df_eg_train[name]['cl3d_PU'] = np.abs(df_eg_train[name].genpart_pt - df_eg_train[name].cl3d_pt_c0)
  df_eg_train[name]['target'] = df_eg_train[name].genpart_pt/df_eg_train[name].cl3d_pt_c0

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
  df_eg_train[name]['cl3d_pt_c1'] = df_eg_train[name]['cl3d_pt_c0']-(((model_c1[name].coef_)*df_eg_train[name]['cl3d_abseta'])+(model_c1[name].intercept_))
  df_eg_train[name]['cl3d_response_c1'] = df_eg_train[name].cl3d_pt_c1 / df_eg_train[name].genpart_pt

###########

# Training calibration 2
print('Training eta calibration with GradientBoostingRegressor')

features = ['cl3d_abseta', 'cl3d_n',
  'cl3d_showerlength', 'cl3d_coreshowerlength', 
  'cl3d_firstlayer', 'cl3d_maxlayer', 
  'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean',
  'cl3d_hoe', 'cl3d_meanz', 
  'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 
  'cl3d_ntc67', 'cl3d_ntc90']

model_c2 = {}
GBR_feature_importance = [[None for _ in range(len(features))] for _ in range(len(df_eg_train))]

for name in df_eg_train:

  input_c2 = df_eg_train[name][features]
  target_c2 = df_eg_train[name]['target']
  model_c2[name] = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=2, random_state=0, loss='huber').fit(input_c2, target_c2)

for name in df_eg_train:

  with open(file_out_model_c2[name], 'wb') as f:
    pickle.dump(model_c2[name], f)

for name in df_eg_train:

  df_eg_train[name]['cl3d_c2'] = model_c2[name].predict(df_eg_train[name][features])
  df_eg_train[name]['cl3d_pt_c2'] = df_eg_train[name].cl3d_pt_c0 * (df_eg_train[name].cl3d_c2)
  df_eg_train[name]['cl3d_response_c2'] = df_eg_train[name].cl3d_pt_c2 / df_eg_train[name].genpart_pt
  GBR_feature_importance[name] = model_c2[name].feature_importances_

###########

# Training calibration 2 (xgboost)
print('Training eta calibration with xgboost')

inputs = ['cl3d_abseta','cl3d_n','cl3d_showerlength','cl3d_coreshowerlength',
   'cl3d_firstlayer', 'cl3d_maxlayer',
   'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean',
   'cl3d_hoe', 'cl3d_meanz',
   'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90',
   'cl3d_ntc67', 'cl3d_ntc90']

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

  output = 'target' 
  model_c3[name], df_eg_train[name]['output_c3']= train_xgb(df_eg_train[name], inputs, output, param, test_fraction=0.4)
  ##cv_results = xgb.cv(param)
  ##print(cv_results)
    
for name in df_eg_train:

  with open(file_out_model_c3[name], 'wb') as f:
    pickle.dump(model_c3[name], f)

for name in df_eg_train:

  full = xgb.DMatrix(data=df_eg_train[name][inputs], label=df_eg_train[name][output], feature_names=inputs)
  df_eg_train[name]['cl3d_c3'] = model_c3[name].predict(full)
  df_eg_train[name]['cl3d_pt_c3'] = df_eg_train[name].cl3d_pt_c0 * (df_eg_train[name].cl3d_c3)
  df_eg_train[name]['cl3d_response_c3'] = df_eg_train[name].cl3d_pt_c3 / df_eg_train[name].genpart_pt

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

## FEATURE IMPORTANCES XGBOOST

matplotlib.rcParams.update({'font.size': 16})

for name in df_eg_train:

  plt.figure(figsize=(15,10))
  plt.rcdefaults()
  fig, ax = plt.subplots()
  y_pos = np.arange(len(features))
  ax.barh(y_pos, GBR_feature_importance[name], align='center')
  for index, value in enumerate(GBR_feature_importance[name]):
    ax.text(value+.005, index+.25, str(value), color='black')#, fontweight='bold')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(features)
  ax.invert_yaxis()  # labels read top-to-bottom
  ax.set_xlabel('Feature Importance')
  ax.set_title('Gradient Boosting Regressor')
  plt.subplots_adjust(left=0.35, right=0.80, top=0.85, bottom=0.2)
  plt.savefig(plotdir+'GBR_bdt_importances_'+pileup+'_'+clustering+'_'+fe_names[name]+'.png')
  plt.savefig(plotdir+'GBR_bdt_importances_'+pileup+'_'+clustering+'_'+fe_names[name]+'.pdf')

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
  plt.hist((df_eg_train[name].cl3d_response_c2), bins=np.arange(0.0, 1.4, 0.01), label = 'C + GBR', histtype = 'step')
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
    df_effrms_etaC3 = df_eg_train[name].groupby(['bineta']).apply(lambda x: rmseff(x.cl3d_response_c3))
    df_effrms_ptC3 = df_eg_train[name].groupby(['binpt']).apply(lambda x: rmseff(x.cl3d_response_c3)) 
    df_rms_eta = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_Uncorr))
    df_rms_pt = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_Uncorr))
    df_rms_etaC1 = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_c1))
    df_rms_ptC1 = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_c1))
    df_rms_etaC2 = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_c1))
    df_rms_ptC2 = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_c1))
    df_rms_etaC3 = df_eg_train[name].groupby(['bineta']).apply(lambda x: np.std(x.cl3d_response_c1))
    df_rms_ptC3 = df_eg_train[name].groupby(['binpt']).apply(lambda x: np.std(x.cl3d_response_c1))
    if(name==0):
        fig = plt.figure(num='performance',figsize=(32,32))
        plt.title(pileup+'_'+clustering)
    plt.figure(num='performance')
    plt.subplot(441)
    plt.title('Uncorrected')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(442)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(443) 
    plt.errorbar((df_mean_eta.cl3d_abseta), df_effrms_eta/df_mean_eta.cl3d_response_Uncorr, linestyle='-', marker='o',  label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(444) 
    plt.errorbar(np.abs(df_mean_pt.genpart_pt), df_effrms_pt/df_mean_pt.cl3d_response_Uncorr, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(445)
    plt.title('Linear Regression')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(446)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(447)
    plt.errorbar((df_mean_eta.cl3d_abseta),df_effrms_etaC1/df_mean_eta.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(448)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC1/df_mean_pt.cl3d_response_c1, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(449)
    plt.title('Gradient Boosting Regressor')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(4,4,10)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(4,4,11)
    plt.errorbar((df_mean_eta.cl3d_abseta),df_effrms_etaC2/df_mean_eta.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(4,4,12)
    plt.errorbar((df_mean_pt.genpart_pt),df_effrms_ptC2/df_mean_pt.cl3d_response_c2, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)

    plt.subplot(4,4,13)
    plt.title('XGBOOST Regression')
    plt.errorbar((df_mean_eta.cl3d_abseta), df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(4,4,14)
    plt.errorbar((df_mean_pt.genpart_pt), df_mean_pt.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.75,1.05)
    plt.xlabel('$p{_T}^{\gamma}$',fontsize=20)
    plt.ylabel('$p{_T}^{Cl3d}$/$p{_T}^{\gamma}$',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(4,4,15)
    plt.errorbar((df_mean_eta.cl3d_abseta),df_effrms_etaC3/df_mean_eta.cl3d_response_c3, linestyle='-', marker='o', label=fe_names[name])
    plt.ylim(0.01,0.10)
    plt.xlabel('$\eta^\gamma$',fontsize=20)
    plt.ylabel('resolution',fontsize=20)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.subplot(4,4,16)
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

