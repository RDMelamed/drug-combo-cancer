import tables
import sys
import pandas as pd
import numpy as np
import pickle
import pdb
import csv
import datetime
import time
from collections import Counter, defaultdict
from scipy import sparse
import scipy
from itertools import chain
from sklearn.preprocessing import StandardScaler
import subprocess
import os
import json
import glob

import util
dem_length = 7

def chunk_list(trt_h5):
    return [i._v_name for i in trt_h5.walk_groups()][1:]

def node2csr(node, future=False):
    attributes = []
    for attribute in ('data', 'indices', 'indptr', 'shape'):
        attributes.append(getattr(node, attribute).read())
    #if future: ## these should all be ones
    #    attributes[0] = np.ones(attributes[0].shape)
    return sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])

def load_chunk(trt_h5, tn, trt_useids, sparse_index, future=False):
    node = trt_h5.get_node("/" + tn )
    ids = node.den[:,0] #ids = node[:,0]
    sel = np.isin(ids, trt_useids)
    if sel.sum() == 0:
        return None
    #trt_dense = node[:][sel,:]
    trt_dense = node.den[:][sel,:]
    #pdb.set_trace()
    M = node2csr(node)
    M = M[sel,:][:,sparse_index]
    ret = [trt_dense, M]
    if future:
        ret.append(node.future[sel])
    return ret

def get_h5(hisdir, trt_drugid):
    return tables.open_file(util.sparseh5_names(hisdir, trt_drugid),'r')



def write_ids(hisdir, drug, write_to):
    h5 = get_h5(hisdir, drug)
    ids = []
    for tn in chunk_list(h5):
        node = h5.get_node("/" + tn )
        ids.append(node.den[:,0])
    np.savetxt(write_to, np.hstack(ids),fmt="%d")

    

def load_selected(hisdir, trt_drugid, trt_useids, sparse_index):
    trt_h5 = get_h5(hisdir, trt_drugid)
    trt_sparse = []
    trt_dense = []
    #ftsum = np.zeros(30285) #sparse_index.shape[0])
    for tn in chunk_list(trt_h5):
        d = load_chunk(trt_h5, tn, trt_useids, sparse_index)
        if not d:
            continue
        d, s = d
        trt_dense.append(d)
        trt_sparse.append(s)
        #ftsum += sm
    #pdb.set_trace()
    return trt_dense, trt_sparse


def load_outcomes(h5, useids, omax):
    ret = []
    ids_ret = []
    for tn in chunk_list(h5):
        node = h5.get_node("/" + tn )
        ids = node.den[:,0] #ids = node[:,0]
        sel = np.isin(ids, useids)
        if sel.sum() == 0:
            continue
        ids = ids[sel]
        outcomes = node.outcomes[sel]
        ret.append(omat(outcomes,omax))
        ids_ret.append(ids)
    if len(ids_ret)==0:
        return np.zeros(0), np.zeros(0)
    return np.hstack(ids_ret), np.vstack(ret)

chunk_size = 300001

    
def align_to_sparseindex(x, sparse_index):
    x = x[:,np.isin(x[0,:], sparse_index)]
    x = x[:,np.argsort(x[0,:])]
    mycol = np.where(np.isin(sparse_index, x[0,:]))[0]
    return x, mycol


def omat(nodeinfo,omax):
    pdat = []

    for pato in list(nodeinfo):
        pato = list(pato)
        ixy = 1
        outs = [pato[0]]
        ogot = 0
        while ixy < len(pato):
            while pato[ixy] > ogot:
                outs += [0]
                #print("no {:d}, putting 0".format(ogot))                
                ogot += 1
            outs.append(pato[ixy + 1]) #if pato[ixy + 1] > 0 else 0
            #print(ogot, outs[-1], pato[ixy + 1], ixy)
            ogot += 1
            ixy += 2
        #outs = outs + [0]*(omax - ogot)
        pdat.append(outs + [0]*(omax - ogot))
    pdat = np.array(pdat) #pdat
    anyo = np.where(pdat[:,1:] > 0, pdat[:,1:], 5000).min(axis=1).reshape(-1,1)
    anyo = np.where(anyo==5000,0,anyo)
    return np.hstack((pdat,anyo))


#TIME_CHUNK = 25
def censored_sparsemat(hisdir, past_sparse_index, use_ids, this_drug, other_drug,TIME_CHUNK, agg=1,  washout=np.inf, any_washout=np.inf, ret_fut_dense=False, keep_vi = []):
    t0 = time.time()
    voc = pd.read_pickle("../../data/clid.vi.allvocab.pkl")        
    #fut_sparse_index = np.arange(1,voc['vi'].max()+1)
    #fut_made_index = False
    other_vi =  -1 if other_drug==0 else voc.loc[(voc['type']=="rx") &
                                                 (voc['id']==other_drug),
                                                 "vi"].values[0]
    this_vi = voc.loc[(voc['type']=="rx") & (voc['id']==this_drug),"vi"].values[0]
    type_vi = [set(voc.loc[(voc['type']==t)& (voc['ct']>100),"vi"].values)
               for t in ['rx','dx','px']]
    #pdb.set_trace()
    ncol = len(past_sparse_index)

    r = 0
    fut_sparse_elements = defaultdict(int)
    import psutil
    process = psutil.Process(os.getpid())

    h5 = get_h5(hisdir, this_drug)
    
    for tn in chunk_list(h5):
        node = h5.get_node("/" + tn )
        patid = node.den[:,0]
        sel = np.isin(patid, use_ids)
        for future in node.future[sel]:
            future_periods = np.where(np.array(future)==-2)[0]
            for el in set(chain.from_iterable([future[(future_periods[fp]+2):future_periods[fp+1]]
                                               for fp in range(len(future_periods)-1)])):
                fut_sparse_elements[el] += 1
    print("t1: make future index {:1.2f}".format(time.time() - t0))
    if other_drug > 0 and other_vi in fut_sparse_elements:
        fut_sparse_elements.pop(other_vi)
    fut_sparse_elements.pop(0)
    if washout < np.inf: ## shouldn't use presence of drug to predict itself
        fut_sparse_elements.pop(this_vi)
    ct = pd.DataFrame(fut_sparse_elements,index=['ct']).transpose()
    lab = []
    past = []
    dense = []    
    fut_sparse_index_for_fit = np.array(ct.loc[(ct['ct'] > 100),:].index)
    fut_sparse_index = fut_sparse_index_for_fit
    if keep_vi:
        fut_sparse_index = np.array(ct.loc[(ct['ct'] > 100) |
                                       (ct.index.isin(keep_vi)),:].index)
    #pdb.set_trace()
    #sel = np.where(np.isin(fut_elts, fut_sparse_index))[0]
    futmat = {'rows':[], 'cols':[], 'dat':[], 'ncol':len(fut_sparse_index)}            
    print("Censmat: {:d} past elt & {:d} fut elt".format(past_sparse_index.shape[0], fut_sparse_index.shape[0]))
    fut_dense = []
    r = 0
    #washouts = []
    for tn in chunk_list(h5):

        ret = load_chunk(h5, tn, use_ids,
                         past_sparse_index, future=True)
        if not ret:
            continue
        dense_c, past_c, future_c = ret
        c_inds = []
        time_inc = []
        time_size = []
        for row in range(len(future_c)):
            
            #past_dat = np.array(past_c[row,:].todense())
            #past_cols = np.where(past_row >  0)[0]
            #npast_dat =  past_dat[past_cols]
            #den = dense_c[row,:]
            #if dense_c[row,0]==105882114:
            #    pdb.set_trace()
            future = future_c[row]
            if len(future)==0:
                c_inds.append(row)
                time_inc.append(0)
                time_size.append(0)
                fut_dense.append([0,0,0])
                r += 1
                lab += [0] ## person is not censored
                #washouts += [0]
                continue
            future_periods = np.where(np.array(future)==-2)[0]
            if agg > 1:
                future_agg = []
                f_ind = 0
                while f_ind < len(future_periods):
                    chunk_len  = 0
                    chunk_contents = []
                    for toagg in range(agg):
                        if f_ind == len(future_periods):
                            break
                        fstart = future_periods[f_ind]
                        last_chunk = f_ind == len(future_periods)-1
                        chunk_len += future[fstart + 1] if last_chunk else TIME_CHUNK
                        chunk_contents += list(future[fstart + (2 if last_chunk else 2) :(len(future) if last_chunk else future_periods[f_ind+1])])
                        f_ind +=  1

                    delim = [-2]
                    if f_ind >= len(future_periods):  ## for last chunk only...
                        delim.append(chunk_len)
                    future_agg += delim + sorted(set(chunk_contents))
                future = future_agg
                future_periods = np.where(np.array(future)==-2)[0]            
            last_drug1_ago = 0
            last_drug_any_ago = 0
            ### for some people, the index  date is their last date. It's not
            ###   right to remove them...
            for fut in range(len(future_periods)):
                c_inds.append(row)
                time_inc.append(TIME_CHUNK*agg*fut)
                last_chunk = fut >= len(future_periods)-1
                time_size.append(0 if not last_chunk else future[future_periods[fut]+1])

                ### get columns of the future periods co
                fut_elt = future[future_periods[fut]+(2 if last_chunk else 1):
                                 (len(future) if last_chunk else future_periods[fut+1])]
                fut_mycol = np.where(np.isin(fut_sparse_index, fut_elt))[0]
                #fut_x2, fut_mycol = align_to_sparseindex(fut_elt, fut_sparse_index)
                futmat['rows'] += [r]*len(fut_mycol)
                futmat['cols'] += list(fut_mycol)
                futmat['dat'] += [1]*len(fut_mycol) #fut_mycol)
                fdadd = [len(t_vi & set(fut_elt)) for t_vi in type_vi]
                if ret_fut_dense:
                    fut_dense.append(fdadd)
                    if len(fut_dense) < len(c_inds):
                        pdb.set_trace()
                r += 1
                if this_vi not in fut_elt:
                    last_drug1_ago += TIME_CHUNK*agg
                else:
                    last_drug1_ago = 0
                #pdb.set_trace()

                if len(type_vi[0] & set(fut_elt)) == 0:
                    last_drug_any_ago += TIME_CHUNK*agg
                else:
                    last_drug_any_ago = 0

                if (other_vi in fut_elt or
                    last_drug1_ago > washout or
                    last_drug_any_ago > any_washout):
                    #if last_drug1_ago > washout:
                    #    pdb.set_trace()
                    lab += [1]
                    #washouts += [last_drug1_ago]
                    break
                else:
                    lab += [0]
            
        print("CHUNK  {:s}, nrow={:d} {:2.2f}".format(tn, len(lab), process.memory_info().rss/10**9))
                
        dense_c = dense_c[c_inds,:]
        dense_c[:,1] += np.array(time_inc)
        dense_c[:,2] += (np.array(time_inc)/52.0)
        dense_c = np.hstack((np.reshape(dense_c[:,0],(1,-1)).transpose(),
                             np.reshape(np.array(time_size),(1,-1)).transpose(),
                             dense_c[:,1:]))
        past_c = past_c[c_inds,:]
        dense.append(dense_c)
        past.append(past_c)

    #np.savetxt(str(this_drug) + "washouts",np.array(washouts))
    dense = np.vstack(dense)
    past = sparse.vstack(past)
    futmat = sparse.csr_matrix((futmat['dat'],(futmat['rows'],futmat['cols'])),
                              shape=(r,futmat['ncol']))

    t1  = time.time()
    print("END nrow={:d} {:2.2f}, time = {:1.2f} min".format(dense.shape[0], process.memory_info().rss/10**9,(t1 - t0)/60))

    cens_info = pd.DataFrame({'ids':dense[:,0],
                              "censored":lab})

    ## transform week to week SINCE drug
    def offs(w): return list(w - w.min())
    woffs  = pd.DataFrame(dense[:,[0,2]],columns=['ids','week']).groupby("ids")['week'].agg(offs)

    cens_info["interval_start"] = np.hstack([woffs[k] for k in
                                   cens_info['ids'].drop_duplicates()])
    interval_length = np.where(dense[:,1]==0,TIME_CHUNK*agg, dense[:,1])
    cens_info["interval_end"] = cens_info['interval_start'] + interval_length


    ret = [dense, past, futmat, lab, fut_sparse_index, cens_info, interval_length]
    if keep_vi:
        ret.append(fut_sparse_index_for_fit)
    if ret_fut_dense:
        fut_dense = np.array(fut_dense)
        ftn = ['rx','dx','px']
        fut_dense = pd.DataFrame(fut_dense, columns = ftn)
        fut_dense['ids'] = cens_info['ids']
        fut_dense['ilen'] = interval_length        
        #fut_dense = fut_dense.groupby('ids').agg('cumsum')
        #pdb.set_trace()
        ## average number Rx etc
        #fut_dense.loc[:,ftn] = fut_dense.loc[:,ftn]/fut_dense['ilen']
        ret.append((fut_dense.loc[:,ftn].values.transpose()/fut_dense['ilen'].values).transpose())
    return ret
            
    #store_to(node_name)
    #h5tab.close()
    


                
def create_sparse_index(hisdir, drugid, savename):
    sparse_elements = defaultdict(int) ## only needed for the "treated"
    trtid = 0
    pdone = 0
    trt_h5 = get_h5(hisdir, drugid)    
    for tn in chunk_list(trt_h5):
        node = trt_h5.get_node("/" + tn )
        spsum  = np.array(node2csr(node).sum(axis=0))[0,:]
        for i,el in enumerate(spsum):
            sparse_elements[i] += el 

    ct = pd.DataFrame(sparse_elements,index=['ct']).transpose()
    index =list(ct.loc[ct['ct'] > 100,:].index)
    sparse_index = index
    f = open(savename,'wb')
    pickle.dump((ct), f)
    f.close()
