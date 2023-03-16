import pandas as pd
import numpy as np
import pickle
import pdb
import datetime
import time
from collections import Counter, defaultdict
from scipy import sparse
import scipy
import os
import tables
import sys
import glob
import subprocess
import json
from itertools import chain
import multiprocessing as mp

## other utilities
import regression_util as rs
import load_subject_info as his2ft
import util
import create_weights as wt

def one_basedrug(drugid, hisdir,resdir, outcomes,
                 TIME_CHUNK,drug_freq_filter=3000, multi=True,
                 agg=1, drug1_washout=np.inf, drug_any_washout=np.inf,
                 FILT_ERA=0, ctl_outcomes_name='', all_combos=''):
    
    if not hisdir.endswith("/"): hisdir += "/"
    if not resdir.endswith("/"): resdir += "/"

    ### Block 1
    comboname = util.combo_names2(resdir, drugid)

    ## get patients who took drug A
    trtid_file = comboname + "trtid"    

    if not os.path.exists(trtid_file):
        try:
            his2ft.write_ids(hisdir, drugid, trtid_file)
        except OSError: ## this was not done
            return
    trtid = np.loadtxt(trtid_file,dtype=int)
    ### end Block 1

    ### Block 2
    BIGNESS = 300000/max(1,3-agg, (drug1_washout - 20) % (agg*12))
    if drug1_washout == np.inf:
        BIGNESS = 100000
    
    NSAMP = 5
    BFRAC = 1.2
    idfiles =  []
    #pdb.set_trace()
    suff2 = wt.suff(agg, FILT_ERA, drug1_washout,drug_any_washout).strip(".")

    ### split up patients who took drug A into cohorts based on a heuristic so that the person-time matrix is not giant
    if len(trtid) > BFRAC*BIGNESS:
        truesplits = int(np.ceil(len(trtid)/BIGNESS))
        trt_splits = min(truesplits, NSAMP)
        trt_split_index = rs.get_splitvec(len(trtid), truesplits)
        for i in range(trt_splits):
            #fname = comboname + "spl" + str(i)
            fname = comboname + "spl" + str(i) + suff2 
            if not os.path.exists(fname):
                tid = trtid[trt_split_index==i]
                if tid.shape[0] > 800000:
                    tid = np.random.choice(tid, size=800000, replace=False)
                np.savetxt(fname,tid)
            else:
                idfiles = sorted(glob.glob(comboname + "spl*" +suff2))
                break
            idfiles.append(fname)
    else:
        idfiles = [comboname + 'trtid']
    print("splits ",",".join(idfiles))        
    del trtid
    ### end Block 2

    ### Block 3
    ## matching codes to their meanings
    voc = pd.read_pickle("../../data/clid.vi.allvocab.pkl")
    
    ## outcomes = dictionary of {disease name:[medical code list] }
    outcomes = sorted(pickle.load(open(outcomes,'rb')).keys())

        
    ### negative control outcomes
    ### ctl_outcomes = dict of {'outcome name': [list of vocab ids]}
    ctl_outcomes = json.loads(open(ctl_outcomes_name).read())
    ctl_outcomes_name = ctl_outcomes_name.split(".")[0]
    ctlo = sorted(ctl_outcomes.keys())
    outcomes_list =  [ctl_outcomes[i] for i in ctlo]
    ### end Block 3
    
    ### Block 4
    ### get all of the possible health codes as features
    sparse_ix_name = util.sparse_index_name(comboname)
    if not os.path.exists(sparse_ix_name):
        his2ft.create_sparse_index(hisdir, drugid, sparse_ix_name)    

    past_sparse_index = util.get_sparseindex(comboname)

    rx_vi = set(voc.loc[voc['type']=='rx','vi'].values)
    outcome_vi_keep = sorted(chain.from_iterable(outcomes_list))

    ### all these people have no history of cancer but may have histoy of the CTL outcomes
    ### thus we need to make sure we are recording their past CTL outcomes
    past_inp = np.array(sorted(set(outcome_vi_keep) | set(past_sparse_index)))
    ### end Block 4

    ### for each of our cohorts (or all patients who took the drug, if not giant)
    for runi,fname in enumerate(idfiles):
        ### Block 5
        runi  = str(runi) +  "."
        run_save = comboname + runi  + suff2 + "."
        nopastuct = pd.DataFrame()
        d2counts = run_save + ".counts.txt"
        complete_log = comboname + "complete_log" + ctl_outcomes_name
        complete = [] if not os.path.exists(complete_log) else open(complete_log).read().strip().split("\n")
        
        ret = his2ft.censored_sparsemat(hisdir,
                            past_inp, np.loadtxt(fname), drugid,
                            0, TIME_CHUNK, agg=agg,
                                        washout=drug1_washout, any_washout=drug_any_washout,
                                        keep_vi = outcome_vi_keep)
        [dense, past, fut, lab, fut_sparse_ix, cens_info, ch_len, fut_ix_fit] = ret[:8]

        past.data = np.clip(past.data,0,1) #past = past > 0
        todo = set(fut_sparse_ix) & rx_vi ## this is for obtaining the drug2's
        #### end Block 5


        ### Block 6
        if not os.path.exists(d2counts):
            sel = np.where(np.isin(fut_sparse_ix, list(rx_vi)))[0]
            rxvin = fut_sparse_ix[sel]

            res = {}
            for i in range(len(sel)):
                if not rxvin[i] in past_sparse_index:
                    continue
                idd=  pd.DataFrame({'ids':cens_info['ids'],
                                    'past':np.array(past[:,np.where(past_inp==rxvin[i])[0]].todense())[:,0],
                                    'drug2':np.array(fut[:,sel[i]].todense())[:,0]}).groupby('ids').agg(max)


                res[rxvin[i]] = ((idd['past']==0) & (idd['drug2'] > 0)).sum() #idd.sum().values[0]


            nopastuct = pd.DataFrame(res,index=['ct']).transpose().sort_values('ct',ascending=False)
            #nopastuct.to_pickle(run_save + ".counts.pkl")
            nopastuct.to_csv(d2counts,sep="\t")
        #pdb.set_trace()
        todo = list(nopastuct.loc[nopastuct['ct'] > drug_freq_filter].index)
        ### end Block 6
        
        if len(todo) == 0:
            continue

        ### Block 7
        ### we included some elements in history just to get if person has history
        ### of disease... might include sparse not good things, remove for regression
        ### just keep if person (row) has history of outcome
        history_of = np.array([np.array(past[:,np.where(np.isin(past_inp,ovi))[0]].sum(axis=1)>0)[:,0]
                        for ovi in outcomes_list]).transpose()

        past = past[:,np.where(np.isin(past_inp, past_sparse_index))[0]]
        ### end block 7

        ### Block 8
        ## we also included outcomes in the follow-up features, though they might not be common enough to use for
        ##   regression-- we want to keep the info about when the outcomes happen, but remove it from the features for regression
        ## we get the future-incidences which correspond to our particular outcomes (outcomes_list = list of the Vocab Ids VI)
        x  = pd.DataFrame(np.transpose(np.array([np.array(fut[:,np.where(np.isin(fut_sparse_ix,ovi))[0]].sum(axis=1)>0)[:,0]
                            for ovi in outcomes_list])),
                          columns = ctlo,dtype=int)
        x = pd.concat((cens_info['ids'],x),axis=1)
        y = x.groupby('ids').agg(np.cumsum)
        y = pd.concat((x['ids'],y),axis=1)
        z = y.groupby('ids').agg(np.cumsum)
        fut_inc = z.mask(z > 1, other = -1)
        fut_inc = fut_inc.mask(history_of, other=-1)
        del history_of, x, y, z 

        fut = fut[:,np.isin(fut_sparse_ix, fut_ix_fit)]
        fut_sparse_ix = fut_ix_fit
        lab = np.array(lab)
        ### end Block 8

        ### begin Block 9
        discont_file = run_save + "bz2"
        #pdb.set_trace()

        ### make discsontinuation censoring: censor those who have not taken drug A recently
        ### to avoid confounding, make model of probabiklity of censoring to weight person-time
        if not os.path.exists(discont_file):
            if (lab==1).sum() > 0:
                wt.p2weight(cens_info, lab, dense, past, fut,
                            ch_len/(TIME_CHUNK*agg),
                            run_save, past_sparse_index, fut_sparse_ix,
                            p='discont')
            else:
                cens_info['discont.cum_wt']= np.ones(cens_info.shape[0])
            #### save/load here
            ### cancer outcomes is stored in the data structure, just need to recover names..
            ids, outcome_inc = his2ft.load_outcomes(his2ft.get_h5(hisdir, drugid), np.loadtxt(fname), len(outcomes))
            outcome_inc = pd.DataFrame(outcome_inc,index=np.hstack(ids),
                                       columns=['deenroll'] + outcomes + ['any_outcome'])
            ### now, we have a matrix with the rows = person-time uncensored, columns = censoring weights + outcomes 
            cens_info = wt.add_outcomes(outcome_inc, cens_info, TIME_CHUNK*agg)
            cens_info = cens_info.drop(['Cancer_history','Unspecified_Cancer','Obs_uspected_neoplasm','Secondary_Malignant_Neoplasm','any_outcome'],axis=1)
            cens_info.to_csv(discont_file,sep="\t",header=True,compression='bz2')
        else:
            cens_info = pd.read_csv(discont_file,index_col=0,sep="\t")
        ### end block 9

        ### block 10
        ## remove censored (discontinued)time points
        uncensored = np.where(lab==0)[0]
        cens_info = cens_info.iloc[uncensored,:]
        ch_len = ch_len[uncensored]
        dense =  dense[uncensored,:];  past=past[uncensored,:]; fut = fut[uncensored,:]
        fut_inc = fut_inc.iloc[uncensored,:] #history_of =  history_of[uncensored,:];
        ### end block 10
            
        #pdb.set_trace()        
        
        
        for drug2 in todo:
            if sel_todo and not drug2 in sel_todo:
                continue

            ### Block 11
            drug2f = run_save + str(drug2) + "." 
            if drug2f in complete:
                print("...have:", run_save + str(drug2))
                continue

            combhis = cens_info.copy()            
            todrop = set(['censored','discont.den','discont.num','discont.single_wt']) & set(combhis.columns)
            if todrop:
                combhis = combhis.drop(todrop,axis=1).copy()

            ## add in if take drug2 in this period
            combhis['drug2'] = np.array(fut[:,np.where(fut_sparse_ix==drug2)[0]].todense())[:,0]

            init_weights = drug2f + "cens_trt.bz2" ## will have drug 2 init treat, init drug2-weights
            #print("creating: " + init_weights)
            if (combhis.groupby('ids')['drug2'].agg(max)).sum() < 1000 or (os.path.exists(drug2f + "trials.bz2") and os.path.exists(drug2f + ctl_outcomes_name + ".bz2")):
                pdb.set_trace()
                #ok = check_files(drug2f, ctl_outcomes_name)
                ok = True
                if ok:
                    with open(complete_log,'a') as f:
                        #f.write(runi + str(drug2) + "\n")
                        f.write(drug2f + "\n")
                
                with open(complete_log,'a') as f:
                    f.write(runi + str(drug2) + "\n")
                #pdb.set_trace()
                continue
            ### pastdrug = took d2 *before* d1. then any affect is ?? becuase some people might have taken d2 only before, others take it before & after
            ### prevdrug = any time point after init d2. These points will be kept in as follow-up time but not weighted or fit in model
            prevdrug = combhis.groupby('ids')['drug2'].transform(lambda  x: list(np.cumsum(np.array([0]+list(x.values[:-1])))))
            pastdrug = np.array(past[:,np.where(past_sparse_index==drug2)[0][0]].todense())[:,0]

            ### only want to fit model on isntances with no previous drug2
            ### -- this means not before drug 1; and not after drug2 started
            sel =  np.where((prevdrug==0) & (pastdrug==0))[0]
            comb_init = []

            comb_init = combhis.iloc[sel,:].copy()
            ### end Block 11

            ### Block 12
            fut_columns =  fut_sparse_ix!=drug2
            print("Weights for "+ drug2f  + " {:d}x{:d}".format(comb_init.shape[0],fut.shape[1] +past.shape[1]))
            fut_do = fut[sel,:][:,np.where(fut_columns)[0]]
            wt.p2weight(comb_init, comb_init['drug2'].values, dense[sel,:], past[sel,:],
                        fut_do,
                        ch_len[sel]/(TIME_CHUNK*agg),
                         drug2f, past_sparse_index, fut_sparse_ix[fut_columns],p="drug2")
            comb_init = comb_init.drop(['drug2.den','drug2.num'],axis=1)
            ### end Block 12

            ### Block 13
            ### fixing the fact that there will be rows that were dropped for
            ### fitting the model.
            init_weight = np.ones(combhis.shape[0])
            init_weight[sel] = comb_init['drug2.single_wt']
            print("Trials for "+ drug2f,combhis.shape)
            combhis['init_weight'] = init_weight

            ## check someone who innitiated after time 1 but before  last time
            combhis['drug2.cum_wt'] = combhis.groupby('ids')['init_weight'].agg('cumprod')
            del comb_init
            combhis = combhis.loc[pastdrug==0,:]
            ### end Block 13

            ### Block 14
            ### expanding into trials of person-time
            trials = []
            dsel = []
            treated = []
            ## for each person, obtain their trials, up through the initiation of drug B each time is a trial
            for pat in set(combhis['ids']):
                patrows = np.where(combhis['ids']==pat)[0]
                patdat = combhis.iloc[patrows,:]
                for trial in range(patdat.shape[0]):
                    nr = patdat.shape[0] - trial
                    trials += [trial]*nr
                    trial_treated = patdat['drug2'].iloc[trial]
                    treated += [trial_treated]*nr
                    dsel += list(patrows[trial:patdat.shape[0]])
                    ### no longer "eligible" if treated  here
                    if trial_treated:
                        break
            print("Expanding "+ drug2f +" to ",len(dsel))

            ### end Block 14

            ### Block 15
            fut_inc.loc[pastdrug==0,:].iloc[dsel,:].to_csv(drug2f + ctl_outcomes_name + ".bz2", sep="\t",header=True,compression="bz2", index=False)            
            trial_file = drug2f + "trials.bz2"
            if not os.path.exists(trial_file):
                #pdb.set_trace()
                combhis = combhis.iloc[dsel,:]
                combhis['trial'] = trials
                combhis['treatment'] = treated
                if drug1_washout==np.inf:
                    combhis['totwt'] = combhis['drug2.cum_wt']
                else:
                    combhis['totwt'] = combhis['discont.cum_wt']*combhis['drug2.cum_wt']

                to_drop = set(['drug2','discont.cum_wt','drug2.cum_wt','init_weight']) & set(combhis.columns)
                combhis.drop(to_drop,axis=1).to_csv(trial_file,sep="\t",header=True,compression="bz2",index=False)
            ### end Block 15
            print("Finished:",drug2f)
            with open(complete_log,'a') as f:
                #f.write(runi + str(drug2) + "\n")
                f.write(drug2f + "\n")
            

    '''
    if os.path.exists(runname + "complete_log"):
        finished = open(runname+"complete_log").read().strip().split("\n")
j        remaining = set(dodict.keys()) - set([drugid])  - set([int(i.replace("complete",""))
                                                               for i in finished if i != "complete"])
        if len(remaining)== 0:
            print("COMPLETED-SKIPPING:" + runname )
            return
        dodict = {k:dodict[k] for k in list(remaining) + [drugid]}
    '''
def check_files(drug2f, ctl_outcomes_name):
    ok = True
    try:
        x = pd.read_csv(drug2f + "trials.bz2",sep="\t")
    except:
        os.remove(drug2f + "trials.bz2")
        print(drug2f , "NOT OK!!")
        return False
    try:
        x = pd.read_csv(drug2f + ctl_outcomes_name + ".bz2",sep="\t")
    except:
        os.remove(drug2f +  ctl_outcomes_name + ".bz2")
        return False
    return True
    
def run_r(Q):
    for doname in iter(Q.get, None):
        tocall = ("Rscript --vanilla /project2/melamed/wrk/iptw/code/matchweight/_eff.R " + outc_fnames + " " + 
                        ("single_cross_section"  if single_outcomes else "multi_cross_section")
                        + " " + str(False) )
        print(tocall)
        subprocess.call(tocall,shell=True)        
        runone(paste(cdir, filename,sep=""))    
def run_single(hisdir, resdir, drug, ctl_outcomes_name="ctloutcomes_match.json", dff=3000,
               agg=2, drug1_washout=20):

    foutcomes = "outcomes_no_nonmel.pkl"
    TIME_CHUNK = 12
    fut_den_feat=False
    #drug1_washout=20
    drug_any_washout=np.inf
    FILT_ERA = 0
    combos_do =  '' #"pairs_todo.json"
    if drug1_washout > 20 or agg < 2:
        dff = 2000
        #combos_do = "pairs_filter1.json"
        combos_do = "pairs_filter_june_cancers.json"
    import json
    print("start-start ", drug)
    #dff = 3000 if len(sys.argv)<3 else int(sys.argv[2])
    #ctloutcomes = json.loads(open("ctloutcomes.json").read())
    one_basedrug(int(drug),hisdir,resdir, foutcomes,TIME_CHUNK,agg=agg, drug1_washout=drug1_washout,ctl_outcomes_name=ctl_outcomes_name, drug_freq_filter=dff, all_combos=combos_do)
    #pdb.set_trace()
    
    midfix = "agg-" + str(agg) +  ".drug1-" + str(drug1_washout)
    tocall = "Rscript --vanilla run_eff.R " + str(drug) + " " + midfix  + " " + ctl_outcomes_name.split(".")[0]
    subprocess.call(tocall,shell=True)
    trfiles = glob.glob(hisdir + "Combo2." + str(drug) + "/*" + midfix + "*trials.bz2")
    wtfiles = glob.glob(hisdir + "Combo2." + str(drug) + "/*" + midfix + "*trials.bz2.wt.txt")
    trfiles = [ i + ".wt.txt" for i in trfiles]
    if set(trfiles) - set(wtfiles):
        return False
    else:
        print("done-done ", drug)
        return True



'''
import drug_combo_background 
for i in glob.glob("hisagain-12/ctodo.curr.6*"): 
     print(i) 
     dr = open(i).read().strip() 
     drug_combo_background.run_single("hisagain-12/", dr) 
     os.remove(i)
'''        


def run_many(subdo, agg=2, drug1_washout=20):
    hisdir = "hisagain-12/"
    if not hisdir.endswith("/"): hisdir+="/"
    def readtodo(todo):
        if not os.path.exists(todo):
            return []
        return open(todo).read().strip().split("\n")
    suff= ('agg-' + str(agg) if agg!=2 else '') + ('d1-' + str(drug1_washout) if drug1_washout!=20 else '')
    suff = suff if not suff else "." + suff
    
    todo = readtodo(hisdir + "ctodo" + suff)
    finished = readtodo(hisdir + "ctodo.finished" + suff)
    todo_updated = set(todo) - set(finished)
    #pdb.set_trace()
    if not len(todo_updated) == len(todo):
        with open(hisdir + "ctodo" + suff,'w') as f:
            f.write("\n".join(todo_updated) + "\n")
    #curlog = glob.glob("logs/clog*")
    #subdo = '0' if not curlog else str(1+max([int(i.split(".")[1]) for i in curlog]))
    for currtodo in glob.glob(hisdir + "ctodo.curr*"+suff):
        todo_updated = todo_updated - set(readtodo(currtodo))
    subprocess.call("touch logs/clog." + subdo + suff, shell=True)
    thisdo = list(todo_updated)[:min(len(todo_updated), 1)]
    thisdo = thisdo if isinstance(thisdo, list) else [thisdo]    
    print("ctodo.curr." +  subdo +  " DOIN:  " + "\n".join(thisdo), 'agg='+ str(agg) + ' drug1_washout='+str(drug1_washout))
    with open(hisdir + "ctodo.curr." + subdo + suff,'w') as f:
        f.write("\n".join(thisdo) + "\n")
    sq = str(subprocess.check_output("squeue --user=melamed", shell=True,universal_newlines=True)).strip().split("\n") 
    sq = [[j.strip().replace("['","").replace("']","") for j in str(i).split(" ") if len(j.strip()) > 0] for i in sq]
    sq = pd.DataFrame(sq[1:],columns=sq[0])
    #pdb.set_trace()
    jobs = list(sq.loc[sq['NAME'].str.contains("run_com"),"JOBID"])
    #pdb.set_trace()
    if len(jobs) < 45 and len(set(todo_updated)- set(thisdo)) > 0:
        subprocess.call("sbatch run_combo.sh " + str(agg) + " " + str(drug1_washout), shell=True)

    for ddo in thisdo:
        with open("logs/clog." + subdo + suff,'a') as f:
            f.write("started "  + str(ddo) + "\n")
        finish = run_single(hisdir, ddo, agg=int(agg), drug1_washout=int(drug1_washout) if np.float(drug1_washout) < np.float('inf') else np.float('inf'))
        if finish:
            with open(hisdir + "ctodo.finished" + suff,'a') as f:
                f.write(ddo + "\n") # #sys.argv[1])
            with open("logs/clog." + subdo + suff,'a') as f:
                f.write("finished "  + str(ddo) + "\n")
        
if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_many(sys.argv[1])
    else:
        run_many(sys.argv[1],sys.argv[2], sys.argv[3])
    '''
    one_basedrug(int(sys.argv[1]),"hisagain-12/","outcomes_no_nonmel.pkl",12, Q,
                 drug1_washout=30 if len(sys.argv) < 3 else np.float(sys.argv[2]),
                 agg= 2 if len(sys.argv) < 4 else int(sys.argv[3]),                 
                 drug_in='' if len(sys.argv) < 5 else sys.argv[4])
    '''
