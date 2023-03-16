
import pandas as pd
import numpy as np
import pickle
import pdb
import csv
from sklearn.metrics import roc_auc_score
from scipy import sparse
import scipy
from sklearn.preprocessing import MaxAbsScaler
import sys
import os
#sys.path.append("../12.21_allcode_ctlmatch")
import tables
import bz2
from patsy import dmatrix
import subprocess


import load_subject_info as his2ft
import regression_util as rs
import util
def get_mod(mod, whichdo=''):
    return whichdo if whichdo else (list(mod['preds'].keys())[0]
                                        if len(mod['preds'])==1
                                        else mod['xval'].mean(axis=1).idxmax())

def ipw(mod, whichdo=''):
    whichmod = get_mod(mod, whichdo)
    ps = mod['preds'][whichmod]
    
    return pd.Series(np.clip((mod['lab']*(mod['lab'].mean()/np.clip(ps,0,1)) +
                      (1-mod['lab'])*((1-mod['lab'].mean())/np.clip(1-ps,10**-6,1))),
                             0,10000),
                     index = mod['ids'])



def get_weights(XS, lab, full_interval, savename,
                past_sparse_index, fut_sparse_ix=[], dnames=[]):
    den = {}
    if lab.sum() < 10:
        return np.zeros(XS.shape[0])
    print("get_weights:",savename)
    if os.path.exists(savename): # or os.stat(savename).st_size == 0:
        try:
            den = pickle.load(open(savename,'rb'))
        except:
            try:
                den = pickle.load(bz2.open(savename,'rb'))
            except:
                den = {}
    if not den or den['scaler'].scale_.shape[0] != XS.shape[1]:
        den = rs.cross_val(XS, lab, 5,iter=15000000,alphas=10**np.arange(-4.0,-2.0),
                           l1s=[.3,.2], fit_sel=full_interval==1)
        den = {'xval':den[4],'mods':den[0],
               'scaler':den[1],#'preds':den[3],
               'past_ix':past_sparse_index, 'fut_ix':fut_sparse_ix,
               'dnames':dnames}
        with bz2.open(savename, 'wb') as f:
            pickle.dump(den, f)

        #f = open(savename,'wb')
        #pickle.dump(den,f)
        #f.close()
            
    setting = den['xval'].mean(axis=1).idxmax()
    preds = den['mods'][setting].predict_proba(den['scaler'].transform(XS))[:,1] if not 'preds' in den else den['preds'][setting]
    #pdb.set_trace()
    preds = full_interval*preds #np.where(full_interval==0, TIME_CHUNK, full_interval)/TIME_CHUNK*preds
    #np.clip(preds, 10**-8,1)
    
    return preds

def get_splined_dense(dense, save_to, p, cens_info):
    spline_info = rs.get_spline_info(dense[:,2:])
    splinify = rs.info2splines(spline_info)
    formula = rs.get_formula(spline_info)
    #pdb.set_trace()
    spline_timesince = rs.spline_dict(pd.DataFrame(cens_info['interval_start']))
    
    dnames = ['sex']
    try:
        dadd = dmatrix(formula, rs.get_patsydict(spline_info, dense[:min(10000,dense.shape[0]),2:])).design_info.column_names
        dnames.extend(dadd)
    except ValueError:
        formula = formula.replace('cc(month,knots=[1,','cc(month,knots=[2,')
        dadd = dmatrix(formula, rs.get_patsydict(spline_info, dense[:min(10000,dense.shape[0]),2:])).design_info.column_names
        dnames.extend(dadd)
    dense = np.hstack((splinify(dense[:,2:]), spline_timesince))
    #time_since = cens_info['interval_start'].values.reshape(-1,1)                      
    #dense = np.hstack((splinify(dense[:,2:]), time_since, time_since**2, time_since**3))
    #dnames += ['time','time**2','time**3']
    dnames += spline_timesince.design_info.column_names
    #del spline_timesince
    with open(save_to + p+ ".dnames",'w') as f:
        f.write("\n".join(dnames) + "\n")
    return dense, dnames

def calc_single_cum(cens_info,p, lab):
    cens_info[p+'single_wt'] = (1-lab)*np.clip(1-cens_info[p+'num'],10**-5,1)/np.clip(1-cens_info[p+'den'],10**-5,1) + \
                             lab*np.clip(cens_info[p + 'num'],10**-5,1)/np.clip(cens_info[p+'den'], 10**-5,1)
    if pd.isnull(cens_info[p + 'single_wt'].sum())>0:
        pdb.set_trace()
    cens_info[p+'cum_wt'] = cens_info.groupby('ids')[p+'single_wt'].agg('cumprod')

def cciweight(cens_info, lab, dense, pastcci, futcci, interval, save_to, p):
    if p and not p.endswith("."):
        p += "."
    
    dense, dnames = get_splined_dense(dense, save_to, p, cens_info)
    cens_info[p+'den'] = get_weights(sparse.hstack((dense,pastcci,futcci), format='csr'), lab, interval,
                                   save_to + p+"den.pkl",
                                     None, None, dnames)
    cens_info[p+'num'] = get_weights(sparse.hstack((dense,pastcci), format='csr'),
                                   lab, interval,
                                   save_to + p+ "num.pkl",
                                     None, [], dnames)
    calc_single_cum(cens_info, p, lab)

def agoexp(hisft, wid=360):
    xx = np.array(hisft.data)
    hisft.data = np.exp(-1*(xx)**2/wid)
    return hisft
    
    
def p2weight(cens_info, lab, dense, spmat, futmat, interval,
             save_to, past_sparse_index, fut_sparse_ix,
             p=''):
    if p and not p.endswith("."):
        p += "."
    #if os.path.exists(num_file):
    #    cens_info[p+'den'] = get_weights(1,np.ones(100),3, den_file, 4,5,6)
    #    cens_info[p+'num'] = get_weights(1,np.ones(100),3, num_file, 4,5,6)
    #    make_weights(cens_info, p, lab)
    #    return
    spline_info = rs.get_spline_info(dense[:,2:])
    splinify = rs.info2splines(spline_info)
    formula = rs.get_formula(spline_info)
    #pdb.set_trace()
    spline_timesince = rs.spline_dict(pd.DataFrame(cens_info['interval_start']))
    
    dnames = ['sex']
    try:
        dadd = dmatrix(formula, rs.get_patsydict(spline_info, dense[:min(10000,dense.shape[0]),2:])).design_info.column_names
        dnames.extend(dadd)
    except ValueError:
        formula = formula.replace('cc(month,knots=[1,','cc(month,knots=[2,')
        dadd = dmatrix(formula, rs.get_patsydict(spline_info, dense[:min(10000,dense.shape[0]),2:])).design_info.column_names
        dnames.extend(dadd)
    dense = np.hstack((splinify(dense[:,2:]), spline_timesince))
    #time_since = cens_info['interval_start'].values.reshape(-1,1)                      
    #dense = np.hstack((splinify(dense[:,2:]), time_since, time_since**2, time_since**3))
    #dnames += ['time','time**2','time**3']
    dnames += spline_timesince.design_info.column_names
    #del spline_timesince
    with open(save_to + p+ ".dnames",'w') as f:
        f.write("\n".join(dnames) + "\n")
                       #, cens_info['interval_start'].values.reshape(-1,1)**2))
    #full_interval = dense[:,1]
    #pdb.set_trace()

    keep = np.array((spmat > 0).sum(axis=0))[0,:]
    keep = (keep > 100) & (keep < .7*spmat.shape[0])
    if (~keep).sum() > 0:
        spmat = spmat[:,keep]
        past_sparse_index = past_sparse_index[keep]
        print("FILTERING ultrasparse MSM data:",(~keep).sum(), "->",spmat.shape, ' for ', save_to)
    spmat = agoexp(spmat)

    cens_info[p+'den'] = get_weights(sparse.hstack((dense,spmat,futmat), format='csr'), lab, interval,
                                   save_to + p+"den.pkl",
                                     past_sparse_index, fut_sparse_ix, dnames)
    cens_info[p+'num'] = get_weights(sparse.hstack((dense,spmat), format='csr'),
                                   lab, interval, save_to + p+ "num.pkl",
                                     past_sparse_index, [], dnames)  
    #return pd.Series(np.clip((mod['lab']*(mod['lab'].mean()/np.clip(ps,0,1)) +
    #                  (1-mod['lab'])*((1-mod['lab'].mean())/np.clip(1-ps,10**-6,1))),
    #                         0,10000),
    cens_info[p+'single_wt'] = (1-lab)*np.clip(1-cens_info[p+'num'],10**-5,1)/np.clip(1-cens_info[p+'den'],10**-5,1) + \
                             lab*np.clip(cens_info[p + 'num'],10**-5,1)/np.clip(cens_info[p+'den'], 10**-5,1)
    if pd.isnull(cens_info[p + 'single_wt'].sum())>0:
        pdb.set_trace()
    cens_info[p+'cum_wt'] = cens_info.groupby('ids')[p+'single_wt'].agg('cumprod')
    

def add_outcomes(outcomes, cens_info, TIME_CHUNK):
    to_drop = set(outcomes.columns) & set(['ipw','label','deenroll'])
    if to_drop:
        outcomes = outcomes.drop(to_drop, axis=1)
    def cmat(week):
        patid = week.name
        cancs = outcomes.loc[patid,:].values
        week = np.tile(week, (cancs.shape[0],1)).transpose()
        inchunk = np.array((cancs >= week) & (cancs < week + TIME_CHUNK) & (cancs > 0), dtype=int)
        befchunk = np.array((cancs < week) & (cancs > 0), dtype=int)
        return inchunk - befchunk

    gb = cens_info.groupby('ids')['interval_start'].apply(cmat)    
    gb = np.vstack([gb[k] for k in cens_info['ids'].drop_duplicates()])
    return pd.concat((cens_info,
                      pd.DataFrame(gb,columns=outcomes.columns)),axis=1)

def suff(agg, FILT_ERA, drug1_washout, drug_any_washout=np.inf):
    #pdb.set_trace()
    suff2 =  (".agg-" + str(agg) ) +  (".filt-" +  str(FILT_ERA) if FILT_ERA > 0 else '')
    #if drug1_washout < np.inf:
    suff2 +=  ".drug1-" + str(drug1_washout)
    if drug_any_washout < np.inf:
        suff2 +=  ".drugAny-" + str(drug_any_washout)
        
    return suff2

def censoring_weights(hisdir, outname, drug1, drug2, outcomes_sorted,trt, TIME_CHUNK,
                      agg=1, FILT_ERA=0, drug1_washout=np.inf):
    save_suff = (".trt" if drug1==trt else ".ctl")
    suff2 =  suff(agg,FILT_ERA, drug1_washout)
    savebz2 =outname + save_suff + suff2+  ".censwt.bz2"
    print("Starting: ", savebz2)
    if os.path.exists(savebz2):
        try:
            _ =  pd.read_csv(savebz2,sep="\t")
            print("exists: ", savebz2)
            return suff2
        except EOFError:
            print("removing partial: ", savebz2)            
            os.remove(savebz2)
    drug_ids = np.loadtxt(outname + ".ids" + save_suff)
    past_sparse_index = util.get_sparseindex(hisdir, trt)
    _, my_sparse = his2ft.load_selected(hisdir, drug1, drug_ids, past_sparse_index)    
    my_sparse = sparse.vstack(my_sparse, format='csr')
    keep = np.array((my_sparse > 0).sum(axis=0))[0,:]
    keep = (keep > 50) & (keep < .7*my_sparse.shape[0])
    del _, my_sparse
    past_sparse_index = past_sparse_index[keep]
    
    (dense, spmat, futmat, lab, fut_sparse_ix,
     cens_info, interval_length) = his2ft.censored_sparsemat(hisdir,  past_sparse_index,
                                                             np.loadtxt(outname + ".ids" + save_suff), drug1, drug2, TIME_CHUNK, agg=agg, washout=drug1_washout)
    if agg > 1:
        TIME_CHUNK = TIME_CHUNK*agg
    lab = np.array(lab)

    cens_info["as_treated"] = cens_info['interval_end'] if drug1==trt else np.zeros(cens_info.shape[0])
    if FILT_ERA >  0:
        sel = np.where(cens_info['interval_start']  < FILT_ERA)[0]
        cens_info = cens_info.iloc[sel,:]
        dense = dense[sel,:]
        spmat = spmat[sel,:]
        futmat = futmat[sel,:]
        lab = lab[sel]


    p2weight(cens_info, lab, dense, spmat, futmat, interval_length/TIME_CHUNK,
             outname + save_suff +suff2, past_sparse_index, fut_sparse_ix)
    outcomes = pd.DataFrame()
    try:
        outcomes = pd.read_csv(outname + ".iptw",sep="\t")
    except: ### zipped!
        outcomes = pd.read_csv(outname + ".iptw",sep="\t",compression='bz2')
    outcomes = outcomes.loc[outcomes['label']==int(drug1==trt),:].set_index("id")
    cens_info = add_outcomes(outcomes, cens_info, TIME_CHUNK)

    cens_info = cens_info.loc[cens_info['censored']==0,:] #.drop('lab',axis=1)
    print("SAVING: ", savebz2,  cens_info.shape)
    
    todrop = set(['as_treated', 'den', 'num', 'single_wt']) & set(cens_info.columns)
    if len(todrop) > 0:
        cens_info  = cens_info.drop(todrop,axis=1)
    

    cens_info.to_csv(outname + save_suff +suff2+ ".censwt.bz2",sep="\t",header=True,compression='bz2')
    return suff2
