#!/usr/bin/env python
# coding: utf-8
#### Preprocessing script to select events from the HGCAL TPG ntuples. This selects events with generated particle matched to a 3D cluster within a threshold radius. The script stores only events which are inside the threshold radius. Events having matched clusters are stored with 'best_match' = True.                        
## Uses param.py file to give the inputs and path to output directory  

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import uproot
from datetime import date
import optparse
import h5py

workdir=os.getcwd()
algo_name=['F8T0','F8T0th0','F8T0th20','F8T0dr10','F8T0dr20','F8T0dr30','F8T0dr40','F8T0dr50','F8T0dr60','F8T0dr70','F8T0dr80','F8T0dr90','F8T0dr200','F8T1','F8T1th0','F8T1th20','F8T1dr10','F8T1dr20','F8T1dr30','F8T1dr40','F8T1dr50','F8T1dr60','F8T1dr70','F8T1dr80','F8T1dr90','F8T1dr200','F8T','F8Tth0','F8Tth20','F8Tdr10','F8Tdr20','F8Tdr30','F8Tdr40','F8Tdr50','F8Tdr60','F8Tdr70','F8Tdr80','F8Tdr90','F8Tdr200','F8BC','F8BCth0','F8BCth20','F8BCdr10','F8BCdr20','F8BCdr30','F8BCdr40','F8BCdr50','F8BCdr60','F8BCdr70','F8BCdr80','F8BCdr90','F8BCdr200']

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
    branches_cl3d=['event','cl3d_n','cl3d_id','cl3d_pt','cl3d_energy','cl3d_eta','cl3d_phi','cl3d_clusters_n','cl3d_clusters_id','cl3d_showerlength','cl3d_coreshowerlength','cl3d_firstlayer','cl3d_maxlayer','cl3d_seetot','cl3d_seemax','cl3d_spptot','cl3d_sppmax','cl3d_szz','cl3d_srrtot','cl3d_srrmax','cl3d_srrmean','cl3d_emaxe']
    branches_T23=branches_cl3d+['cl3d_bdteg','cl3d_quality']
    branches_cl3dlayer=['cl3d_layer_pt']    
    
    for i,filename in enumerate(files,1):
        tree[0]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxGenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[1]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxth0GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[2]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxth20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[3]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr10GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[4]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[5]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr30GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[6]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr40GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[7]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr50GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[8]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr60GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[9]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr70GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[10]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr80GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[11]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr90GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[12]=uproot.open(filename)['Floatingpoint8Threshold0DummyHistomaxdr200GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[13]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxGenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[14]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxth0GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[15]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxth20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[16]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr10GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[17]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[18]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr30GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[19]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr40GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[20]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr50GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[21]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr60GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[22]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr70GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[23]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr80GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[24]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr90GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[25]=uproot.open(filename)['Floatingpoint8Threshold1DummyHistomaxdr200GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[26]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxGenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[27]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxth0GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[28]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxth20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[29]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr10GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[30]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[31]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr30GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[32]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr40GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[33]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr50GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[34]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr60GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[35]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr70GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[36]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr80GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[37]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr90GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[38]=uproot.open(filename)['Floatingpoint8ThresholdDummyHistomaxdr200GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[39]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxGenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[40]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxth0GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[41]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxth20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[42]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr10GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[43]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr20GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[44]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr30GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[45]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr40GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[46]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr50GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[47]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr60GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[48]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr70GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[49]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr80GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[50]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr90GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        tree[51]=uproot.open(filename)['Floatingpoint8BestchoiceDummyHistomaxdr200GenmatchGenclustersntuple;1/HGCalTriggerNtuple']
        if i==1:
            gen=tree[0].pandas.df(branches_gen,flatten=True)
            algo[0]=tree[0].pandas.df(branches_T23,flatten=True)
            algo[1]=tree[1].pandas.df(branches_cl3d,flatten=True)
            algo[2]=tree[2].pandas.df(branches_cl3d,flatten=True)
            algo[3]=tree[3].pandas.df(branches_cl3d,flatten=True)
            algo[4]=tree[4].pandas.df(branches_cl3d,flatten=True)
            algo[5]=tree[5].pandas.df(branches_cl3d,flatten=True)
            algo[6]=tree[6].pandas.df(branches_cl3d,flatten=True)
            algo[7]=tree[7].pandas.df(branches_cl3d,flatten=True)
            algo[8]=tree[8].pandas.df(branches_cl3d,flatten=True)
            algo[9]=tree[9].pandas.df(branches_cl3d,flatten=True)
            algo[10]=tree[10].pandas.df(branches_cl3d,flatten=True)
            algo[11]=tree[11].pandas.df(branches_cl3d,flatten=True)
            algo[12]=tree[12].pandas.df(branches_cl3d,flatten=True)
            algo[13]=tree[13].pandas.df(branches_cl3d,flatten=True)
            algo[14]=tree[14].pandas.df(branches_cl3d,flatten=True)
            algo[15]=tree[15].pandas.df(branches_cl3d,flatten=True)
            algo[16]=tree[16].pandas.df(branches_cl3d,flatten=True)
            algo[17]=tree[17].pandas.df(branches_cl3d,flatten=True)
            algo[18]=tree[18].pandas.df(branches_cl3d,flatten=True)
            algo[19]=tree[19].pandas.df(branches_cl3d,flatten=True)
            algo[20]=tree[20].pandas.df(branches_cl3d,flatten=True)
            algo[21]=tree[21].pandas.df(branches_cl3d,flatten=True)
            algo[22]=tree[22].pandas.df(branches_cl3d,flatten=True)
            algo[23]=tree[23].pandas.df(branches_cl3d,flatten=True)
            algo[24]=tree[24].pandas.df(branches_cl3d,flatten=True)
            algo[25]=tree[25].pandas.df(branches_cl3d,flatten=True)
            algo[26]=tree[26].pandas.df(branches_cl3d,flatten=True)
            algo[27]=tree[27].pandas.df(branches_cl3d,flatten=True)
            algo[28]=tree[28].pandas.df(branches_cl3d,flatten=True)
            algo[29]=tree[29].pandas.df(branches_cl3d,flatten=True)
            algo[30]=tree[30].pandas.df(branches_cl3d,flatten=True)
            algo[31]=tree[31].pandas.df(branches_cl3d,flatten=True)
            algo[32]=tree[32].pandas.df(branches_cl3d,flatten=True)
            algo[33]=tree[33].pandas.df(branches_cl3d,flatten=True)
            algo[34]=tree[34].pandas.df(branches_cl3d,flatten=True)
            algo[35]=tree[35].pandas.df(branches_cl3d,flatten=True)
            algo[36]=tree[36].pandas.df(branches_cl3d,flatten=True)
            algo[37]=tree[37].pandas.df(branches_cl3d,flatten=True)
            algo[38]=tree[38].pandas.df(branches_cl3d,flatten=True)
            algo[39]=tree[39].pandas.df(branches_cl3d,flatten=True)
            algo[40]=tree[40].pandas.df(branches_cl3d,flatten=True)
            algo[41]=tree[41].pandas.df(branches_cl3d,flatten=True)
            algo[42]=tree[42].pandas.df(branches_cl3d,flatten=True)
            algo[43]=tree[43].pandas.df(branches_cl3d,flatten=True)
            algo[44]=tree[44].pandas.df(branches_cl3d,flatten=True)
            algo[45]=tree[45].pandas.df(branches_cl3d,flatten=True)
            algo[46]=tree[46].pandas.df(branches_cl3d,flatten=True)
            algo[47]=tree[47].pandas.df(branches_cl3d,flatten=True)
            algo[48]=tree[48].pandas.df(branches_cl3d,flatten=True)
            algo[49]=tree[49].pandas.df(branches_cl3d,flatten=True)
            algo[50]=tree[50].pandas.df(branches_cl3d,flatten=True)
            algo[51]=tree[51].pandas.df(branches_cl3d,flatten=True)
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
            algoL[10]=tree[10].pandas.df(branches_cl3dlayer)
            algoL[11]=tree[11].pandas.df(branches_cl3dlayer)
            algoL[12]=tree[12].pandas.df(branches_cl3dlayer)
            algoL[13]=tree[13].pandas.df(branches_cl3dlayer)
            algoL[14]=tree[14].pandas.df(branches_cl3dlayer)
            algoL[15]=tree[15].pandas.df(branches_cl3dlayer)
            algoL[16]=tree[16].pandas.df(branches_cl3dlayer)
            algoL[17]=tree[17].pandas.df(branches_cl3dlayer)
            algoL[18]=tree[18].pandas.df(branches_cl3dlayer)
            algoL[19]=tree[19].pandas.df(branches_cl3dlayer)
            algoL[20]=tree[20].pandas.df(branches_cl3dlayer)
            algoL[21]=tree[21].pandas.df(branches_cl3dlayer)
            algoL[22]=tree[22].pandas.df(branches_cl3dlayer)
            algoL[23]=tree[23].pandas.df(branches_cl3dlayer)
            algoL[24]=tree[24].pandas.df(branches_cl3dlayer)
            algoL[25]=tree[25].pandas.df(branches_cl3dlayer)
            algoL[26]=tree[26].pandas.df(branches_cl3dlayer)
            algoL[27]=tree[27].pandas.df(branches_cl3dlayer)
            algoL[28]=tree[28].pandas.df(branches_cl3dlayer)
            algoL[29]=tree[29].pandas.df(branches_cl3dlayer)
            algoL[30]=tree[30].pandas.df(branches_cl3dlayer)
            algoL[31]=tree[31].pandas.df(branches_cl3dlayer)
            algoL[32]=tree[32].pandas.df(branches_cl3dlayer)
            algoL[33]=tree[33].pandas.df(branches_cl3dlayer)
            algoL[34]=tree[34].pandas.df(branches_cl3dlayer)
            algoL[35]=tree[35].pandas.df(branches_cl3dlayer)
            algoL[36]=tree[36].pandas.df(branches_cl3dlayer)
            algoL[37]=tree[37].pandas.df(branches_cl3dlayer)
            algoL[38]=tree[38].pandas.df(branches_cl3dlayer)
            algoL[39]=tree[39].pandas.df(branches_cl3dlayer)
            algoL[40]=tree[40].pandas.df(branches_cl3dlayer)
            algoL[41]=tree[41].pandas.df(branches_cl3dlayer)
            algoL[42]=tree[42].pandas.df(branches_cl3dlayer)
            algoL[43]=tree[43].pandas.df(branches_cl3dlayer)
            algoL[44]=tree[44].pandas.df(branches_cl3dlayer)
            algoL[45]=tree[45].pandas.df(branches_cl3dlayer)
            algoL[46]=tree[46].pandas.df(branches_cl3dlayer)
            algoL[47]=tree[47].pandas.df(branches_cl3dlayer)
            algoL[48]=tree[48].pandas.df(branches_cl3dlayer)
            algoL[49]=tree[49].pandas.df(branches_cl3dlayer)
            algoL[50]=tree[50].pandas.df(branches_cl3dlayer)
            algoL[51]=tree[51].pandas.df(branches_cl3dlayer)
        else:
            gen=pd.concat([gen,tree[0].pandas.df(branches_gen,flatten=True)])
            algo[0]=pd.concat([algo[0],tree[0].pandas.df(branches_T23,flatten=True)])
            algoL[0]=pd.concat([algoL[0],tree[0].pandas.df(branches_cl3dlayer)])
            algo[1]=pd.concat([algo[1],tree[1].pandas.df(branches_cl3d,flatten=True)])
            algo[2]=pd.concat([algo[2],tree[2].pandas.df(branches_cl3d,flatten=True)])
            algo[3]=pd.concat([algo[3],tree[3].pandas.df(branches_cl3d,flatten=True)])
            algo[4]=pd.concat([algo[4],tree[4].pandas.df(branches_cl3d,flatten=True)])
            algo[5]=pd.concat([algo[5],tree[5].pandas.df(branches_cl3d,flatten=True)])
            algo[6]=pd.concat([algo[6],tree[6].pandas.df(branches_cl3d,flatten=True)])
            algo[7]=pd.concat([algo[7],tree[7].pandas.df(branches_cl3d,flatten=True)])
            algo[8]=pd.concat([algo[8],tree[8].pandas.df(branches_cl3d,flatten=True)])
            algo[9]=pd.concat([algo[9],tree[9].pandas.df(branches_cl3d,flatten=True)])
            algo[10]=pd.concat([algo[10],tree[10].pandas.df(branches_cl3d,flatten=True)])
            algo[11]=pd.concat([algo[11],tree[11].pandas.df(branches_cl3d,flatten=True)])
            algo[12]=pd.concat([algo[12],tree[12].pandas.df(branches_cl3d,flatten=True)])
            algo[13]=pd.concat([algo[13],tree[13].pandas.df(branches_cl3d,flatten=True)])
            algo[14]=pd.concat([algo[14],tree[14].pandas.df(branches_cl3d,flatten=True)])
            algo[15]=pd.concat([algo[15],tree[15].pandas.df(branches_cl3d,flatten=True)])
            algo[16]=pd.concat([algo[16],tree[16].pandas.df(branches_cl3d,flatten=True)])
            algo[17]=pd.concat([algo[17],tree[17].pandas.df(branches_cl3d,flatten=True)])
            algo[18]=pd.concat([algo[18],tree[18].pandas.df(branches_cl3d,flatten=True)])
            algo[19]=pd.concat([algo[19],tree[19].pandas.df(branches_cl3d,flatten=True)])
            algo[20]=pd.concat([algo[20],tree[20].pandas.df(branches_cl3d,flatten=True)])
            algo[21]=pd.concat([algo[21],tree[21].pandas.df(branches_cl3d,flatten=True)])
            algo[22]=pd.concat([algo[22],tree[22].pandas.df(branches_cl3d,flatten=True)])
            algo[23]=pd.concat([algo[23],tree[23].pandas.df(branches_cl3d,flatten=True)])
            algo[24]=pd.concat([algo[24],tree[24].pandas.df(branches_cl3d,flatten=True)])
            algo[25]=pd.concat([algo[25],tree[25].pandas.df(branches_cl3d,flatten=True)])
            algo[26]=pd.concat([algo[26],tree[26].pandas.df(branches_cl3d,flatten=True)])
            algo[27]=pd.concat([algo[27],tree[27].pandas.df(branches_cl3d,flatten=True)])
            algo[28]=pd.concat([algo[28],tree[28].pandas.df(branches_cl3d,flatten=True)])
            algo[29]=pd.concat([algo[29],tree[29].pandas.df(branches_cl3d,flatten=True)])
            algo[30]=pd.concat([algo[30],tree[30].pandas.df(branches_cl3d,flatten=True)])
            algo[31]=pd.concat([algo[31],tree[31].pandas.df(branches_cl3d,flatten=True)])
            algo[32]=pd.concat([algo[32],tree[32].pandas.df(branches_cl3d,flatten=True)])
            algo[33]=pd.concat([algo[33],tree[33].pandas.df(branches_cl3d,flatten=True)])
            algo[34]=pd.concat([algo[34],tree[34].pandas.df(branches_cl3d,flatten=True)])
            algo[35]=pd.concat([algo[35],tree[35].pandas.df(branches_cl3d,flatten=True)])
            algo[36]=pd.concat([algo[36],tree[36].pandas.df(branches_cl3d,flatten=True)])
            algo[37]=pd.concat([algo[37],tree[37].pandas.df(branches_cl3d,flatten=True)])
            algo[38]=pd.concat([algo[38],tree[38].pandas.df(branches_cl3d,flatten=True)])
            algo[39]=pd.concat([algo[39],tree[39].pandas.df(branches_cl3d,flatten=True)])
            algo[40]=pd.concat([algo[40],tree[40].pandas.df(branches_cl3d,flatten=True)])
            algo[41]=pd.concat([algo[41],tree[41].pandas.df(branches_cl3d,flatten=True)])
            algo[42]=pd.concat([algo[42],tree[42].pandas.df(branches_cl3d,flatten=True)])
            algo[43]=pd.concat([algo[43],tree[43].pandas.df(branches_cl3d,flatten=True)])
            algo[44]=pd.concat([algo[44],tree[44].pandas.df(branches_cl3d,flatten=True)])
            algo[45]=pd.concat([algo[45],tree[45].pandas.df(branches_cl3d,flatten=True)])
            algo[46]=pd.concat([algo[46],tree[46].pandas.df(branches_cl3d,flatten=True)])
            algo[47]=pd.concat([algo[47],tree[47].pandas.df(branches_cl3d,flatten=True)])
            algo[48]=pd.concat([algo[48],tree[48].pandas.df(branches_cl3d,flatten=True)])
            algo[49]=pd.concat([algo[49],tree[49].pandas.df(branches_cl3d,flatten=True)])
            algo[50]=pd.concat([algo[50],tree[50].pandas.df(branches_cl3d,flatten=True)])
            algo[51]=pd.concat([algo[51],tree[51].pandas.df(branches_cl3d,flatten=True)])
            algoL[1]=pd.concat([algoL[1],tree[1].pandas.df(branches_cl3dlayer)])
            algoL[2]=pd.concat([algoL[2],tree[2].pandas.df(branches_cl3dlayer)])
            algoL[3]=pd.concat([algoL[3],tree[3].pandas.df(branches_cl3dlayer)])
            algoL[4]=pd.concat([algoL[4],tree[4].pandas.df(branches_cl3dlayer)])
            algoL[5]=pd.concat([algoL[5],tree[5].pandas.df(branches_cl3dlayer)])
            algoL[6]=pd.concat([algoL[6],tree[6].pandas.df(branches_cl3dlayer)])
            algoL[7]=pd.concat([algoL[7],tree[7].pandas.df(branches_cl3dlayer)])
            algoL[8]=pd.concat([algoL[8],tree[8].pandas.df(branches_cl3dlayer)])
            algoL[9]=pd.concat([algoL[9],tree[9].pandas.df(branches_cl3dlayer)])
            algoL[10]=pd.concat([algoL[10],tree[10].pandas.df(branches_cl3dlayer)])
            algoL[11]=pd.concat([algoL[11],tree[11].pandas.df(branches_cl3dlayer)])
            algoL[12]=pd.concat([algoL[12],tree[12].pandas.df(branches_cl3dlayer)])
            algoL[13]=pd.concat([algoL[13],tree[13].pandas.df(branches_cl3dlayer)])
            algoL[14]=pd.concat([algoL[14],tree[14].pandas.df(branches_cl3dlayer)])
            algoL[15]=pd.concat([algoL[15],tree[15].pandas.df(branches_cl3dlayer)])
            algoL[16]=pd.concat([algoL[16],tree[16].pandas.df(branches_cl3dlayer)])
            algoL[17]=pd.concat([algoL[17],tree[17].pandas.df(branches_cl3dlayer)])
            algoL[18]=pd.concat([algoL[18],tree[18].pandas.df(branches_cl3dlayer)])
            algoL[19]=pd.concat([algoL[19],tree[19].pandas.df(branches_cl3dlayer)])
            algoL[20]=pd.concat([algoL[20],tree[20].pandas.df(branches_cl3dlayer)])
            algoL[21]=pd.concat([algoL[21],tree[21].pandas.df(branches_cl3dlayer)])
            algoL[22]=pd.concat([algoL[22],tree[22].pandas.df(branches_cl3dlayer)])
            algoL[23]=pd.concat([algoL[23],tree[23].pandas.df(branches_cl3dlayer)])
            algoL[24]=pd.concat([algoL[24],tree[24].pandas.df(branches_cl3dlayer)])
            algoL[25]=pd.concat([algoL[25],tree[25].pandas.df(branches_cl3dlayer)])
            algoL[26]=pd.concat([algoL[26],tree[26].pandas.df(branches_cl3dlayer)])
            algoL[27]=pd.concat([algoL[27],tree[27].pandas.df(branches_cl3dlayer)])
            algoL[28]=pd.concat([algoL[28],tree[28].pandas.df(branches_cl3dlayer)])
            algoL[29]=pd.concat([algoL[29],tree[29].pandas.df(branches_cl3dlayer)])
            algoL[30]=pd.concat([algoL[30],tree[30].pandas.df(branches_cl3dlayer)])
            algoL[31]=pd.concat([algoL[31],tree[31].pandas.df(branches_cl3dlayer)])
            algoL[32]=pd.concat([algoL[32],tree[32].pandas.df(branches_cl3dlayer)])
            algoL[33]=pd.concat([algoL[33],tree[33].pandas.df(branches_cl3dlayer)])
            algoL[34]=pd.concat([algoL[34],tree[34].pandas.df(branches_cl3dlayer)])
            algoL[35]=pd.concat([algoL[35],tree[35].pandas.df(branches_cl3dlayer)])
            algoL[36]=pd.concat([algoL[36],tree[36].pandas.df(branches_cl3dlayer)])
            algoL[37]=pd.concat([algoL[37],tree[37].pandas.df(branches_cl3dlayer)])
            algoL[38]=pd.concat([algoL[38],tree[38].pandas.df(branches_cl3dlayer)])
            algoL[39]=pd.concat([algoL[39],tree[39].pandas.df(branches_cl3dlayer)])
            algoL[40]=pd.concat([algoL[40],tree[40].pandas.df(branches_cl3dlayer)])
            algoL[41]=pd.concat([algoL[41],tree[41].pandas.df(branches_cl3dlayer)])
            algoL[42]=pd.concat([algoL[42],tree[42].pandas.df(branches_cl3dlayer)])
            algoL[43]=pd.concat([algoL[43],tree[43].pandas.df(branches_cl3dlayer)])
            algoL[44]=pd.concat([algoL[44],tree[44].pandas.df(branches_cl3dlayer)])
            algoL[45]=pd.concat([algoL[45],tree[45].pandas.df(branches_cl3dlayer)])
            algoL[46]=pd.concat([algoL[46],tree[46].pandas.df(branches_cl3dlayer)])
            algoL[47]=pd.concat([algoL[47],tree[47].pandas.df(branches_cl3dlayer)])
            algoL[48]=pd.concat([algoL[48],tree[48].pandas.df(branches_cl3dlayer)])
            algoL[49]=pd.concat([algoL[49],tree[49].pandas.df(branches_cl3dlayer)])
            algoL[50]=pd.concat([algoL[50],tree[50].pandas.df(branches_cl3dlayer)])
            algoL[51]=pd.concat([algoL[51],tree[51].pandas.df(branches_cl3dlayer)])
        
    return(gen, algo, algoL)

def preprocessing(path, files, savedir,  thr):
    
    gen,algo,algoL=openroot(path, files)
    n_rec={}
    algo_clean={}    

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
        
        #select deltar under thr
        sel=algo_pos_merged['deltar']<=thr
        algo_pos_merged=algo_pos_merged[sel]
        sel=algo_neg_merged['deltar']<=thr
        algo_neg_merged=algo_neg_merged[sel]
        
        #matching
        group=algo_pos_merged.groupby('event')
        n_rec_pos=group['cl3d_pt'].size()
        algo_pos_merged['best_match']=group.apply(matching).array
        group=algo_neg_merged.groupby('event')
        n_rec_neg=group['cl3d_pt'].size()
        algo_neg_merged['best_match']=group.apply(matching).array
        
        #keep only matched clusters 
        sel=algo_pos_merged['best_match']==True
#        algo_pos_merged=algo_pos_merged[sel]
    
        sel=algo_neg_merged['best_match']==True
#        algo_neg_merged=algo_neg_merged[sel]
    
        #remerge with NaN values
        algo_pos_merged=pd.concat([algo_pos_merged, unmatched_pos], sort=False).sort_values('event') 
        algo_neg_merged=pd.concat([algo_neg_merged, unmatched_neg], sort=False).sort_values('event')
        
        n_rec[i]=n_rec_pos.append(n_rec_neg)
        algo_clean[i]=pd.concat([algo_neg_merged,algo_pos_merged], sort=False).sort_values('event')
        
        algo_clean[i]['matches']=algo_clean[i]['matches'].replace(np.nan, True)
 #       algo_clean[i].drop(columns=['best_match'], inplace=True)

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
    param=importlib.import_module('param2')
    path=param.path
    files=param.files
    thr=param.thr
    savedir=param.savedir
    
    preprocessing(path, files, savedir, thr)

