#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import uproot
from datetime import date
import optparse
import h5py

workdir=os.getcwd()
algo_name=['Thr','Thrdr015','Thrdr30','Thrdr50','Thrdr200','BCdcen','BCdcendr015','BCdcendr30','BCdcendr50','BCdcendr200']

def deltar(df):
    df['deta']=df['cl3d_eta']-df['genpart_exeta']
    df['dphi']=np.abs(df['cl3d_phi']-df['genpart_exphi'])
    sel=df['dphi']>np.pi
    df['dphi']-=sel*(2*np.pi)
    return(np.sqrt(df['dphi']*df['dphi']+df['deta']*df['deta']))
    
def matching(event):
    return event.cl3d_pt==event.cl3d_pt.max()

def openroot(path, files):
    os.chdir(path)
    algo={}
    algoL={}
    tree={}
    branches_gen=['event','genpart_eta','genpart_phi','genpart_pt','genpart_energy','genpart_dvx','genpart_dvy','genpart_dvz','genpart_ovx','genpart_ovy','genpart_ovz','genpart_mother','genpart_exphi','genpart_exeta','genpart_exx','genpart_exy','genpart_fbrem','genpart_pid','genpart_gen','genpart_reachedEE','genpart_fromBeamPipe','genpart_posx','genpart_posy','genpart_posz']
    branches_cl3d=['event','cl3d_n','cl3d_id','cl3d_pt','cl3d_energy','cl3d_eta','cl3d_phi','cl3d_clusters_n','cl3d_clusters_id','cl3d_showerlength','cl3d_coreshowerlength','cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot','cl3d_seemax','cl3d_spptot','cl3d_sppmax','cl3d_szz','cl3d_srrtot','cl3d_srrmax','cl3d_srrmean','cl3d_emaxe','cl3d_hoe','cl3d_meanz','cl3d_layer10','cl3d_layer50','cl3d_layer90','cl3d_ntc67','cl3d_ntc90','cl3d_ipt','cl3d_ienergy']
    branches_T23=branches_cl3d+['cl3d_bdteg','cl3d_quality']
    branches_cl3dlayer=['cl3d_layer_pt']    

    for i,filename in enumerate(files,1):
        tree[0]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxGenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[1]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr015GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[2]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr30GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[3]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr50GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[4]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr200GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[5]=uproot.open(filename)['Floatingpoint8BestchoicedecentDummyHistomaxGenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[6]=uproot.open(filename)['Floatingpoint8BestchoicedecentDummyHistomaxdr015GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[7]=uproot.open(filename)['Floatingpoint8BestchoicedecentDummyHistomaxdr30GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[8]=uproot.open(filename)['Floatingpoint8BestchoicedecentDummyHistomaxdr50GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[9]=uproot.open(filename)['Floatingpoint8BestchoicedecentDummyHistomaxdr200GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        if i==1:
            gen=tree[0].pandas.df(branches_gen,flatten=True)
            algo[0]=tree[0].pandas.df(branches_T23,flatten=True)
            algo[1]=tree[1].pandas.df(branches_cl3d,flatten=True)
            algo[2]=tree[2].pandas.df(branches_cl3d,flatten=True)
            algo[3]=tree[3].pandas.df(branches_cl3d,flatten=True)
            algo[4]=tree[4].pandas.df(branches_T23,flatten=True)
            algo[5]=tree[5].pandas.df(branches_cl3d,flatten=True)
            algo[6]=tree[6].pandas.df(branches_cl3d,flatten=True)
            algo[7]=tree[7].pandas.df(branches_cl3d,flatten=True)
            algo[8]=tree[8].pandas.df(branches_T23,flatten=True)
            algo[9]=tree[9].pandas.df(branches_cl3d,flatten=True)
            algoL[0]=tree[0].pandas.df(branches_cl3dlayer)
            algoL[1]=tree[1].pandas.df(branches_cl3dlayer)
            algoL[2]=tree[2].pandas.df(branches_cl3dlayer)
            algoL[3]=tree[3].pandas.df(branches_cl3dlayer)
            algoL[4]=tree[4].pandas.df(branches_cl3dlayer)
            algoL[5]=tree[5].pandas.df(branches_cl3dlayer)
            algoL[6]=tree[6].pandas.df(branches_cl3dlayer)
            algoL[7]=tree[7].pandas.df(branches_cl3dlayer)
            algoL[8]=tree[8].pandas.df(branches_cl3dlayer)
            algoL[9]=tree[9].pandas.df(branches_cl3dlayer)
        else:
            gen=pd.concat([gen,tree[0].pandas.df(branches_gen,flatten=True)])
            algo[0]=pd.concat([algo[0],tree[0].pandas.df(branches_T23,flatten=True)])
            algo[1]=pd.concat([algo[1],tree[1].pandas.df(branches_cl3d,flatten=True)])
            algo[2]=pd.concat([algo[2],tree[2].pandas.df(branches_cl3d,flatten=True)])
            algo[3]=pd.concat([algo[3],tree[3].pandas.df(branches_cl3d,flatten=True)])
            algo[4]=pd.concat([algo[4],tree[4].pandas.df(branches_T23,flatten=True)])
            algo[5]=pd.concat([algo[5],tree[5].pandas.df(branches_cl3d,flatten=True)])
            algo[6]=pd.concat([algo[6],tree[6].pandas.df(branches_cl3d,flatten=True)])
            algo[7]=pd.concat([algo[7],tree[7].pandas.df(branches_cl3d,flatten=True)])
            algo[8]=pd.concat([algo[8],tree[8].pandas.df(branches_T23,flatten=True)])
            algo[9]=pd.concat([algo[9],tree[9].pandas.df(branches_cl3d,flatten=True)])
            algoL[0]=pd.concat([algoL[0],tree[0].pandas.df(branches_cl3dlayer)])
            algoL[1]=pd.concat([algoL[1],tree[1].pandas.df(branches_cl3dlayer)])
            algoL[2]=pd.concat([algoL[2],tree[2].pandas.df(branches_cl3dlayer)])
            algoL[3]=pd.concat([algoL[3],tree[3].pandas.df(branches_cl3dlayer)])
            algoL[4]=pd.concat([algoL[4],tree[4].pandas.df(branches_cl3dlayer)])
            algoL[5]=pd.concat([algoL[5],tree[5].pandas.df(branches_cl3dlayer)])
            algoL[6]=pd.concat([algoL[6],tree[6].pandas.df(branches_cl3dlayer)])
            algoL[7]=pd.concat([algoL[7],tree[7].pandas.df(branches_cl3dlayer)])
            algoL[8]=pd.concat([algoL[8],tree[8].pandas.df(branches_cl3dlayer)])
            algoL[9]=pd.concat([algoL[9],tree[9].pandas.df(branches_cl3dlayer)])
        
    return(gen, algo, algoL)

def preprocessing(path, files, savedir,  thr):
    
    gen,algo,algoL=openroot(path, files)
    n_rec={}
    algo_clean={}
    algo_clean_matched={}    

    #clean gen from particles that are not the originals or didn't reach endcap
    sel=gen['genpart_reachedEE']==2 
    gen_clean=gen[sel]
    sel=gen_clean['genpart_gen']!=-1
    gen_clean=gen_clean[sel]

    #split df_gen_clean in two by eta sign
    sel1=gen_clean['genpart_exeta']<=0
    sel2=gen_clean['genpart_exeta']>0
    gen_neg=gen_clean[sel1]
    gen_pos=gen_clean[sel2]

    gen_pos.set_index('event', inplace=True)
    gen_neg.set_index('event', inplace=True)

    for i in algo:
        print(algo_name[i])
        list=algoL[i]['cl3d_layer_pt'].tolist()
        flattened = [val for sublist in list for val in sublist]
        algo[i]['layer']=flattened

        #split clusters in two by eta sign
        sel1=algo[i]['cl3d_eta']<=0
        sel2=algo[i]['cl3d_eta']>0
        algo_neg=algo[i][sel1]
        algo_pos=algo[i][sel2]
        #set the indices
        algo_pos.set_index('event', inplace=True)
        algo_neg.set_index('event', inplace=True)
        #merging
        algo_pos_merged=gen_pos.join(algo_pos, how='left', rsuffix='_algo')
        algo_neg_merged=gen_neg.join(algo_neg, how='left', rsuffix='_algo')
        #calculate deltar
        algo_pos_merged['deltar']=deltar(algo_pos_merged)
        algo_neg_merged['deltar']=deltar(algo_neg_merged)
        
        #keep the unreconstructed values (NaN)
        sel=pd.isna(algo_pos_merged['deltar']) 
        unmatched_pos=algo_pos_merged[sel]
        sel=pd.isna(algo_neg_merged['deltar'])  
        unmatched_neg=algo_neg_merged[sel]
        unmatched_pos['matches']=False
        unmatched_neg['matches']=False
        print('unmatched events pos',len(unmatched_pos))
        print('unmatched events neg',len(unmatched_neg))
        
        #select deltar under thr
        #sel=algo_pos_merged['deltar']<=thr
        #algo_pos_merged=algo_pos_merged[sel]
        #sel=algo_neg_merged['deltar']<=thr
        #algo_neg_merged=algo_neg_merged[sel]
        sel=algo_pos_merged['deltar']<=thr
        algo_pos_merged_dR=algo_pos_merged[sel]
        sel=algo_neg_merged['deltar']<=thr
        algo_neg_merged_dR=algo_neg_merged[sel]
        
        #matching
        group=algo_pos_merged_dR.groupby('event')
        n_rec_pos=group['cl3d_pt'].size()
        algo_pos_merged_dR['best_match']=group.apply(matching).array
        group=algo_neg_merged_dR.groupby('event')
        n_rec_neg=group['cl3d_pt'].size()
        algo_neg_merged_dR['best_match']=group.apply(matching).array

        algo_pos_merged=pd.concat([algo_pos_merged, algo_pos_merged_dR], sort=False).sort_values('event') 
        algo_neg_merged=pd.concat([algo_neg_merged, algo_neg_merged_dR], sort=False).sort_values('event')
        algo_pos_merged['best_match']=algo_pos_merged['best_match'].replace(np.nan, False)
        algo_neg_merged['best_match']=algo_neg_merged['best_match'].replace(np.nan, False)

        print('algo gen best match pos events',algo_pos_merged['best_match'])
        print('algo gen best match neg events',algo_neg_merged['best_match'])
        
        #keep only matched clusters 
#        sel=algo_pos_merged['best_match']==True
#        algo_pos_merged=algo_pos_merged[sel]
    
#        sel=algo_neg_merged['best_match']==True
#        algo_neg_merged=algo_neg_merged[sel]
    
        #remerge with NaN values
        algo_pos_merged=pd.concat([algo_pos_merged, unmatched_pos], sort=False).sort_values('event') 
        algo_neg_merged=pd.concat([algo_neg_merged, unmatched_neg], sort=False).sort_values('event')
    
        n_rec[i]=n_rec_pos.append(n_rec_neg)
        algo_clean[i]=pd.concat([algo_neg_merged,algo_pos_merged], sort=False).sort_values('event')
        
        algo_clean[i]['matches']=algo_clean[i]['matches'].replace(np.nan, True)
 #       algo_clean[i].drop(columns=['best_match'], inplace=True)

        sel = algo_clean[i]['best_match']==True
        algo_clean_matched[i] = algo_clean[i][sel] 
        print('algo gen all events',algo_clean[i]['genpart_pt'])
        print('algo cl3d all events',algo_clean[i]['cl3d_pt'])
        print('algo gen M all events',algo_clean_matched[i]['genpart_pt'])
        print('algo cl3d M all events',algo_clean_matched[i]['cl3d_pt'])

        #save files to savedir
    os.chdir(savedir)   
    gen_clean.to_hdf('gen_cleanthbc.hdf5',key='df')
    for i in algo:
        algo_clean[i].to_hdf('{}.hdf5'.format(algo_name[i]),key='df')
        
        
if __name__=='__main__':
    parser = optparse.OptionParser()
    parser.add_option("-f","--file",type="string", dest="param_path", help="select the path to the parameters file")
   
    (opt, args) = parser.parse_args()

    param_path=opt.param_path
    import importlib
    import sys
    sys.path.append(param_path)
    param=importlib.import_module('param1')
    path=param.path
    files=param.files
    thr=param.thr
    savedir=param.savedir
    
    preprocessing(path, files, savedir, thr)

