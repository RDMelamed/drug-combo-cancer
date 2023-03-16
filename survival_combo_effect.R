library(survey)
library(survival)

                                        #runmany("hisagain-12/mbo.4140/",patternIn=".*drug1-20.*")
#source("combo_msm_more_ctlo.R")
#cdir = "hisagain-12/Combo.7363/"
#patternIn = "1.agg-2.drug1-20.*"
#ctlname = "ctloutcomes_smaller"
#source("combo_msm_more_ctlo.R")
#runmany(cdir, patternIn, ctlname)

runmany <-  function(cdir,patternIn="",ctlname="ctlout", doUNWT=T){
    bzdo = list.files(path=cdir,pattern=paste(patternIn,"trials.bz2$",sep=""))
    dontdo = list.files(path=cdir,pattern=paste(patternIn,".cci.*trials.bz2$",sep=""))
    bzdo = setdiff(bzdo, dontdo)
    filename = bzdo
    donelog = paste(cdir,"effj22", ctlname,sep="")
    alldone = c()
    cat("log is ",donelog,"\n")
    for (filename in bzdo){
        if(file.exists(donelog)){
            alldone = read.csv(donelog, header=F, stringsAsFactors=F)[,1]
        }
        
        cat(filename , "\n")
        if(filename %in% alldone){
            next
            }
        runone(paste(cdir, filename,sep=""), ctlname, doUNWT)
        alldone = c(alldone, filename)
        cat("DONE filename", filename, " n=",length(alldone),"\n")
        
        write(alldone, file=donelog)
    }
}

oopsread <- function(fn,nrows=-1){
    dat = read.table(fn,header=T,sep="\t",nrows=nrows) #,row.names = 1)
    cat("reading ",fn, ' dim = ', dim(dat),"\n")        
    oops = colnames(dat)=="X"
    if(sum(oops)>0){
        cat("keeping ",sum(!oops),"\n")
        dat = dat[,!oops]
    }
    return(dat)
}

runone <- function(fread, ctlname, doUNWT=T){
    #if(file.exists(paste(fread,".N.txt",sep=""))){ return() }
    resdfs = list()
    #cutz = c(1,3,5 ,10)
    cutz = 5 
    #dat = read.table(fread,header=T,sep="\t",nrow=10)
    dat = oopsread(fread,nrows=10)
    
    intervalcol = c("ids","interval_start","interval_end","treatment",
                    "trial","totwt")
    othercol = c('drug2','discont.cum_wt','drug2.cum_wt','init_weight',
                 'discont.num','discont.den','censored','discont.single_wt')
    weightcol = 'totwt'
    cancers = setdiff(colnames(dat), c(intervalcol, othercol))
    ctlo = gsub("trials",ctlname, fread)
    ctloutcomes = data.frame()
    alloutcomes = setdiff(cancers, c( "Cancer_history", "Obs_uspected_neoplasm","Secondary_Malignant_Neoplasm", "Unspecified_Cancer","any_outcome"))
    cancers = setdiff(cancers, c( "Cancer_history", "Obs_uspected_neoplasm","Secondary_Malignant_Neoplasm", "Unspecified_Cancer","any_outcome"))
    if(file.exists(ctlo)){
        ctloutcomes = oopsread(ctlo, nrows=10) #read.table(ctlo,header=T,sep="\t",nrow=10)
        alloutcomes = c(cancers,  colnames(ctloutcomes))
    }
    column_names = c("surv","survse","events","N","Nt","Ntevent")
    wtres = paste(fread, '.wt.txt',sep="")
    unwtres = paste(fread, '.unwt.txt',sep="")
    cat("outcomes:",alloutcomes,"mat=",dim(matrix(nrow=length(alloutcomes),ncol=length(column_names))),"\n")
    result_weighted = data.frame(matrix(nrow=length(alloutcomes),ncol=length(column_names)))
    rownames(result_weighted)=alloutcomes
    colnames(result_weighted)=column_names
    result_unweighted = data.frame(matrix(nrow=length(alloutcomes),ncol=length(column_names)))
    rownames(result_unweighted)=alloutcomes    
    colnames(result_unweighted)=column_names
    #,colnames=column_names)
    if(file.exists(wtres)){
        result_weighted = read.table(wtres, sep="\t")
        if(doUNWT){
            result_unweighted = read.table(unwtres, sep="\t")
        }
        newoc = setdiff(alloutcomes, rownames(result_weighted))

        if( length(newoc) > 0){
	    #cat(length(newoc), ncol=length(column_names),"..")
            to_add = data.frame(matrix(nrow=length(newoc), ncol=length(column_names)))
	    #cat("\nnewoc:",newoc)
            rownames(to_add) = newoc
	    #cat("\nxolnames:",column_names)
            colnames(to_add) = column_names
	    cat(dim(result_weighted), dim(result_unweighted), dim(to_add),"..")
            result_weighted = rbind(result_weighted, to_add)
            if(doUNWT){
                result_unweighted = rbind(result_unweighted, to_add)
            }
        }                        
    }
    if(sum(is.na(result_weighted$Ntevent)) == 0){ #[,ncol(resdfs[['eff']])]))==0){
        cat("nothing left to do, skipping ",wtres,"\n")
        return(result_weighted)
    }
    dat = oopsread(fread)
    if(file.exists(ctlo)){
        cat("reading ",ctlo,"\n")
        ctloutcomes = oopsread(ctlo)
        dat = cbind(dat, ctloutcomes)
        
        rm(ctloutcomes)
        }

    cat("total shape ",dim(dat),"\n")
    nc = colSums(dat[dat$trial==0,alloutcomes]==1)
    N = length(unique(dat$ids))
    Nt = length(unique(dat[dat$treatment==1,'ids']))    

    cat("making ",wtres,"\n")

    ### for all outcomes...
    for (cancer in alloutcomes){
            if(!is.na(result_weighted[cancer,'Ntevent'])){
                #cat("skipping ", cancer, cu)
                next
            }            
        
            cat(cancer,"\n")

            ### obtain the person-time where the person is at risk of outcome (has not had past outcome)
            cancdat = dat[dat[,cancer] >= 0, c(intervalcol, cancer)]
                                        #cancdat = t0[t0[,cancer] >= 0, c(intervalcol, cancer)]
            if(sum(cancdat[cancdat$trial==0,cancer]==1) < 50){
                next
            }

            ### clip the weights, set up the data frame for the survival analysis
            cancdat$interval_start = cancdat$interval_start - 1
            colnames(cancdat)[colnames(cancdat)==cancer] = "cancer"
            cancdat$clipwt = ifelse(cancdat[,weightcol] > cutz , cutz,
                                    cancdat[,weightcol])

            ### store info about number of individuals, number with outcome, number treated with outcoe
            fit = 0
            nc = length(unique(cancdat[cancdat$cancer==1,'ids']))
            Ntevent = length(unique(cancdat[ cancdat$cancer == 1 & cancdat$treatment== 1,'ids']))

            ### do unweighted analysis for comparison-- identical but no weights
            if(doUNWT){
                fit  = coxph(Surv(interval_start, interval_end, cancer) ~ treatment + cluster(ids),
                         data = cancdat)
                result_unweighted[cancer,] = c(coef(fit), SE(fit),  nc, N, Nt, Ntevent)
                                        #colnames(tun) =  column_names
                write.table(result_unweighted, unwtres,sep="\t")
            }

            ### weigthed analysis 
            fit  = coxph(Surv(interval_start, interval_end, cancer) ~ treatment + cluster(ids),
                         data = cancdat, weights = clipwt)

            ### obtain weighted SE/effect
            result_weighted[cancer,] = c(coef(fit), SE(fit),  nc, N, Nt, Ntevent)

            write.table(result_weighted, wtres,sep="\t")

    }
}

