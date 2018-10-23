#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:20:03 2018

@author: theoestienne
"""
import pandas as pd
from sklearn import metrics
import numpy as np
import SimpleITK as sitk
import os

#%%



def sein_metrics(reference_path, prediction_path):
    
    pred = pd.read_csv(prediction_path, index_col=0)
    ref = pd.read_csv(reference_path, index_col=0)
    
    assert pred.shape[0] == ref.shape[0]
    assert pred.shape[1] == 18
    
    pred = pred.loc[ref.index]
    pred = pred.fillna(0.5)
    
    # Malin
    y_true= ref['Malin']
    y_pred = pred['Malin']
    
    auc_malin = metrics.roc_auc_score(y_true, y_pred)
    
    # Tissu glandulaire 
    y_true = ref['Type de lesion'] == 'Tissu glandulaire'
    y_pred = pred['Tissu glandulaire']
    
    auc_tissu = metrics.roc_auc_score(y_true, y_pred)
        
    # Carcinome canalaire infiltrant
    y_true = ref['Type de lesion'] == 'Carcinome canalaire infiltrant'
    y_pred = pred['Carcinome canalaire infiltrant']
    
    auc_carcinome = metrics.roc_auc_score(y_true, y_pred)
    
    # Autres Benins
    autres_benins = ['Adenose sclerosante', 'Autre lesion proliferante',
    'Cicatrice radiaire', 'Fibroadenome','Galactophorite', 
    'Hyperplasie canalaire sans atypie', 'Kyste', 'PASH',
    'Papillome', 'cytosteatonecrose',
    'ganglio intra-mammaire']
    
    y_true = ref['Type de lesion'].isin(autres_benins)
    y_pred = pred[autres_benins].sum(axis=1)
    
    auc_autre_benins = metrics.roc_auc_score(y_true, y_pred)
    
    # Autres malins
    autres_malins = ['Carcinome lobulaire infiltrant', 'Cancer triple negatif', 
                      'Carcinome intracanalaire', 'Carcinome mucineux']
    
    y_true = ref['Type de lesion'].isin(autres_malins)
    y_pred = pred[autres_malins].sum(axis=1)
    
    auc_autres_malins = metrics.roc_auc_score(y_true, y_pred)
    
    total_score = 0.6 * auc_malin + 0.4/4 * (auc_tissu + auc_carcinome + auc_autre_benins + auc_autres_malins)
    
    return total_score

#%%
def thyroide_metrics(reference_path, prediction_path):
    
    pred = pd.read_csv(prediction_path, index_col=0)
    ref = pd.read_csv(reference_path, index_col=0)
    
    assert pred.shape[0] == ref.shape[0]
    assert pred.shape[1] == 1
    
    
    pred = pred.loc[ref.index]
    pred = pred.fillna(0.5)
    
    y_true = ref['anormale']
    y_pred = pred['anormale']
    
    auc = metrics.roc_auc_score(y_true, y_pred)
    
    return auc