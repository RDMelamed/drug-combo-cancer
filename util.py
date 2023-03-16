import pandas as pd
import numpy as np
import pickle
import glob
import os


def get_sparseindex(comboname):
    elct = pickle.load(open(sparse_index_name(comboname),'rb'))
    SPFT_CUT = 100
    sparse_index =np.array(sorted(list(elct.loc[elct['ct'] > SPFT_CUT,:].index)),
                           dtype = int)
    if sparse_index.shape[0] == 0:
        sparse_index =np.array(sorted(list(elct.loc[elct['ct'] > 10,:].index)),
                           dtype = int)
    print("get_sparseindex: deleting zero")
    sparse_index = np.delete(sparse_index,0)
    return sparse_index


def sparse_index_name(combodir):
    return combodir + "sparse_index.pkl"

def combo_names2(hisdir, drugid):
    runname = hisdir + 'Combo2.' + str(drugid) +"/"
    if not os.path.exists(runname):
        os.mkdir(runname)
    return runname

def sparseh5_names(hisdir, drugid):
    spdir = hisdir + "/sparsemat/"
    return spdir + str(drugid) + ".h5"
