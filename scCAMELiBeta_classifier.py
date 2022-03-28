
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import pandas as pd
import pickle as pickle
from scipy.spatial.distance import cdist, pdist, squareform
#import backspinpy
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from sklearn import preprocessing
import matplotlib.patches as mpatches
import datetime
import seaborn as sns
import pandas as pd
import pickle as pickle
from scipy.spatial.distance import cdist, pdist, squareform
#import backspinpy
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from sklearn import preprocessing
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

#from skorch.callbacks import Callback
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier


def TransSpeciesGeneName(dfm, dictfilename, path):
    # dfm=dfpfc, dictfilename=dictfilename, path=path

    with open('%s%s' % (path, dictfilename), 'rb') as f:
        mouse2human_dict = pickle.load(f, encoding='ASCII')
    # dfm=df_f
    # convert the mouse gene name in the human gene name
    mouse_translable = [i for i in dfm.index if i in mouse2human_dict]
    dfm = dfm.loc[mouse_translable, :]
    dfm.index = [mouse2human_dict[i] for i in dfm.index]
    return dfm


def prefilter(datax, filename, path):
    # df_f = dfpfc, filename = filename, path = path
    print("CamelRunning_Prefilter......")
    dropgene = open('%s%s' % (path, filename)).read().split('\n')
    other_genes = ['HBG1', 'HBA1', 'HBA2', 'HBE1', 'HBZ', 'BLVRB', 'S100A6', 'SNAR-E', 'SNAR-A13_loc1',
                   'SNAR-C1_loc1', 'SNAR-A1_loc2', 'SNAR-A8_loc1', 'SNAR-C1_loc2', 'SNAR-A2_loc2', 'SNAR-C4',
                   'SNAR-A12_loc1', 'SNAR-C3', 'SNAR-C1_loc3', 'SNAR-G2', 'SNAR-G1', 'SNAR-A11_loc9',
                   'SNAR-A6_loc3', 'SNAR-A14_loc7', 'SNAR-A6_loc5',
                   'SNAR-A10_loc6', 'SNAR-A5_loc9', 'SNAR-A14_loc3', 'SNAR-A9_loc9', 'SNAR-A11_loc7',
                   'SNAR-B1_loc1', 'SNAR-B1_loc2', 'SNAR-D', 'SNAR-F']
    dropgene.extend(other_genes)
    dropgene = list(set(dropgene))
    df_f = pd.DataFrame(datax.X.T, index=datax.var.index, columns=datax.obs.index)

    df_f = df_f.loc[~np.in1d(df_f.index, dropgene)].astype(float)
    df_f = df_f.loc[np.sum(df_f >= 1, 1) >= 5, :]  # is at least 1 in 5 cells
    df_f = df_f.loc[np.sum(df_f >= 2, 1) >= 2, :]  # is at least 2 in 2 cells
    df_f = df_f.loc[np.sum(df_f >= 3, 1) >= 1, :]  # is at least 3 in 1 cells
    ftlist=np.in1d(datax.var.index, df_f.index.tolist())
    datax.var["Filter1"]=ftlist
    print("CamelRunning_Prefilter......Finished")
    return datax


def MVgenes(datax, wt=3, markerlist=[],plotfig=False, Xlow=-8.5, Xhigh=6.5, Ylow=-2, Yhigh=6.5,alphaValue=0.2, sValue=10,
           fig_args={'figsize': (8, 8), 'facecolor': 'white', 'edgecolor': 'white'}):
    # plotShow default= Ture
    print("CamelRunning_MVgenes......")
    df_f = pd.DataFrame(datax.X.T, index=datax.var.index, columns=datax.obs.index)
    df_f=df_f.loc[datax.var["Filter1"] > 0]
    mu = df_f.mean(1).values
    sigma = df_f.std(1, ddof=1).values
    cv = sigma / mu
    score, mu_linspace, cv_fit, params = CV_Mean(mu, cv, 'SVR', svr_gamma=0.003)
    mu_sorted = mu[np.argsort(score)[::-1]]
    cv_sorted = cv[np.argsort(score)[::-1]]
    y = cv_fit.tolist()
    x = mu_linspace.tolist()
    pars = thrscount(x, y)

    thrs = 0
    for i in range(len(np.log2(mu_sorted) > 0)):
        if i == 0:
            if func(np.log2(cv_sorted[i]), pars[0], pars[1], pars[2]) < np.log2(mu_sorted[i]):
                thrs = thrs + 1
        else:
            if np.log2(cv_sorted[i]) < np.log2(cv_sorted[i - 1]):
                if func(np.log2(cv_sorted[i]), pars[0], pars[1], pars[2]) < np.log2(mu_sorted[i]):
                    thrs = thrs + 1
    thrs = min(max(thrs*wt, 1000), 5000)
    # thrs=2210
    MVlist = df_f.iloc[np.argsort(score)[::-1], :].iloc[:thrs, :].index
    MVlist=list(set(MVlist.tolist()+markerlist))
    ftlist2 = np.in1d(datax.var.index, MVlist)
    datax.var["MVgene"] = ftlist2
    #mu = mu, cv = cv, mu_sorted = mu_sorted, cv_sorted = cv_sorted, thrs = thrs,
    #mu_linspace = mu_linspace, cv_fit = cv_fit,
    #Xlow = -8.5, Xhigh = 6.5, Ylow = -2, Yhigh = 6.5, alphaValue = 0.2, sValue = 10,
    if plotfig==True:
        fig = plt.figure(**fig_args)
        ax = fig.add_subplot(111)
        ax.scatter(np.log2(mu), np.log2(cv), marker='o', edgecolor='none', alpha=0.1, s=5)

        ax.scatter(np.log2(mu_sorted[thrs:]), np.log2(cv_sorted[thrs:]), marker='o', edgecolor='none', alpha=alphaValue, s=sValue,
                   c='r')
        # x.plot(mu_linspace, cv_fit*1.1,'-k', linewidth=1, label='$FitCurve$')
        # plot(linspace(-9,7), -0.5*linspace(-9,7), '-r', label='$Poisson$')
        plt.ylabel('log2 CV')
        plt.xlabel('log2 mean')
        ax.grid(alpha=0.3)
        plt.xlim(Xlow, Xhigh)
        plt.ylim(Ylow, Yhigh)
        ax.legend(loc=1, fontsize=15)
        plt.gca().set_aspect(1.2)
        plt.grid(False)
    print("CamelRunning_MVgenes......Finished")
    return datax, score,  thrs, ax

def CV_Mean(mu, cv, fit_method='SVR', svr_gamma=0.003, x0=[0.5, 0.5], verbose=False):
    ### modified from BackSPIN, (GioeleLa Manno, et al., 2016, PMID: 27716510 )

    log2_m = np.log2(mu)
    log2_cv = np.log2(cv)

    if len(mu) > 1000 and 'bin' in fit_method:
        # histogram with 30 bins
        n, xi = histogram(log2_m, 30)
        med_n = percentile(n, 50)
        for i in range(0, len(n)):
            # index of genes within the ith bin
            ind = where((log2_m >= xi[i]) & (log2_m < xi[i + 1]))[0].astype(int)
            if len(ind) > med_n:
                # Downsample if count is more than median
                ind = ind[random.permutation(len(ind))]
                ind = ind[:len(ind) - int(med_n)]
                mask = ones(len(log2_m), dtype=bool)
                mask[ind] = False
                log2_m = log2_m[mask]
                log2_cv = log2_cv[mask]
            elif (around(med_n / len(ind)) > 1) and (len(ind) > 5):
                # Duplicate if count is less than median
                log2_m = r_[log2_m, tile(log2_m[ind], int(round(med_n / len(ind)) - 1))]
                log2_cv = r_[log2_cv, tile(log2_cv[ind], int(round(med_n / len(ind)) - 1))]
    else:
        if 'bin' in fit_method:
            print('More than 1000 input feature needed for bin correction.')
        pass

    if 'SVR' in fit_method:
        try:
            from sklearn.svm import SVR
            if svr_gamma == 'auto':
                svr_gamma = 1000. / len(mu)
            # Fit the Support Vector Regression
            clf = SVR(gamma=svr_gamma)
            clf.fit(log2_m[:, np.newaxis], log2_cv)
            fitted_fun = clf.predict
            score = np.log2(cv) - fitted_fun(np.log2(mu)[:, np.newaxis])
            params = None
            # The coordinates of the fitted curve
            mu_linspace = np.linspace(min(log2_m), max(log2_m))
            cv_fit = fitted_fun(mu_linspace[:, np.newaxis])
            return score, mu_linspace, cv_fit, params

        except ImportError:
            if verbose:
                print('SVR fit requires scikit-learn python library. Using exponential instead.')
            if 'bin' in fit_method:
                return fit_CV(mu, cv, fit_method='binExp', x0=x0)
            else:
                return fit_CV(mu, cv, fit_method='Exp', x0=x0)
    elif 'Exp' in fit_method:
        from scipy.optimize import minimize
        # Define the objective function to fit (least squares)
        fun = lambda x, log2_m, log2_cv: sum(abs(log2((2. ** log2_m) ** (-x[0]) + x[1]) - log2_cv))
        # Fit using Nelder-Mead algorythm
        optimization = minimize(fun, x0, args=(log2_m, log2_cv), method='Nelder-Mead')
        params = optimization.x
        # The fitted function
        fitted_fun = lambda log_mu: log2((2. ** log_mu) ** (-params[0]) + params[1])
        # Score is the relative position with respect of the fitted curve
        score = np.log2(cv) - fitted_fun(np.log2(mu))
        # The coordinates of the fitted curve
        mu_linspace = np.linspace(min(log2_m), max(log2_m))
        cv_fit = fitted_fun(mu_linspace)
        return score, mu_linspace, cv_fit, params



def thrscount(x, y):
    from scipy.optimize import curve_fit
    from scipy.stats.distributions import t
    initial_guess = [10, 0 - x[-1], y[-1]]
    pars, pcov = curve_fit(func, x, y, p0=initial_guess)
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)

    n = len(y)  # number of data points
    p = len(pars)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t.ppf(1.0 - alpha / 2., dof)
    for i, j, var in zip(range(n), pars, np.diag(pcov)):
        sigma = var ** 0.5
        #print('p{0}: {1} [{2}  {3}]'.format(i, j,
        #                                    j - sigma * tval,
        #                                    j + sigma * tval))
    return pars

def func(x, a, b, c):
    'nonlinear function in a and b to fit to data'
    return a * x * x + b * x + c



def prediction(datax, mcolor_dict,net,learninggroup="train", radarplot=False, fontsizeValue=35,
              datarefplot=None,
               ncolnm=1, bbValue=(1.1, 1.05)):
    #mwanted_order = mwanted_order, mclasses_names = mclasses_names, mprotogruop = dfpfcclus.loc["Cluster"].values,
    #mdf_train_set = mdf_train_set, figsizeV = 18, mtrain_index = mtrain_index, net = net, mreorder_ix = mreorder_ix,
    #mcolor_dict = refcolor_dict, learninggroup = "test"



    if  learninggroup=="train":
        mdf_train_set = pd.DataFrame(datax.obsm["train_set_values"].T, index=datax.uns["train_set_gene"],
                                     columns=datax.obs.index)
        mtrain_index = datax.obs["mtrain_index"].values
        mwanted_order = datax.uns["mwanted_order"]
        mclasses_names = datax.uns["mclasses_names"]
        mprotogruop = datax.obs["Cluster"].values
        dfpfcclus = datax.obs["Cluster"]
        mreorder_ix = [list(mclasses_names).index(i) for i in mwanted_order]
        mbool00 = np.in1d( mclasses_names[mtrain_index],  mwanted_order )
        if (np.sum(mcolor_dict ==None)!=0) or (np.sum(mcolor_dict.index.isin(mwanted_order))!=len(mwanted_order)):
            mcolor_dict={}
            for item in mwanted_order:
                mcolor_dict[item] = (np.array(random.sample(range(0, 999), 3)) / 1000).tolist()
        else:
            mcolor_dict = mcolor_dict.map(lambda x: list(map(lambda y: y/255., x)))
        #color_dict
        #mcolor_dict = mcolor_dict.map(lambda x: list(map(lambda y: y/255., x)))
        #rcParams['savefig.dpi'] = 500
        #mnewcolors = array(list(mcolor_dict[mprotogruop].values))
        normalizer = 0.9*mdf_train_set.values.max(1)[:,np.newaxis]
        ####
        # must be np.float32
        ####
        refdataLR=net.predict_proba((mdf_train_set.values/ normalizer).astype(np.float32).T)

        todaytime=f"{datetime.datetime.now():%Y%m%d%I%M%p}"

        dataRef= refdataLR[:,mreorder_ix]
        mreordername=[]
        for i in mreorder_ix:
            mreordername.append(list(mclasses_names)[i])
        dfprobCL=pd.DataFrame(dataRef*100, index=mdf_train_set.columns,columns=mreordername)
        #dfnewcl=pd.DataFrame(array([xtest,ytest]).T, index=mdf_train_set.columns)
        datax.obsm["Celltype_Score"]=dfprobCL.values
        datax.uns["Celltype_Score_RefCellType"]=dfprobCL.columns.tolist()
        datax.uns["Celltype_OrderNumber"]=mreorder_ix
        if radarplot ==True:
            axm, dfclRef = RadarVisualization(refdataLR=refdataLR, mreorder_ix=mreorder_ix,
                                              #fontsizeValue=fontsizeValue,
                                              Ncolm=ncolnm, bbValue=bbValue,
                                                        mtrain_index=mtrain_index,
                                                        mclasses_names=mclasses_names,
                                              mcolor_dict=mcolor_dict,
                                              dataclpn=None, learninggroup=learninggroup,
                                                        mdf_train_set=mdf_train_set,
                                                        mwanted_order=mwanted_order,
                                                        mprotogruop=mprotogruop,
                                              fontsizeValue=int(100/int(len(mwanted_order)**0.5))
                                              )
            datax.obsm["CelltypeScoreCoordinates"]=dfclRef.values
        return datax

    elif learninggroup=="test":
        mdf_train_set = pd.DataFrame(datax.obsm["test_set_values"].T, index=datax.uns["train_set_gene"],
                                     columns=datax.obs.index)
        mtrain_index = datax.obs["mtrain_index"].values
        mwanted_order = datax.uns["mwanted_order"]
        mclasses_names = datax.uns["mclasses_names"]
        mprotogruop = datax.obs["Cluster"].values
        dfpfcclus = datax.obs["Cluster"]
        mreorder_ix = [list(mwanted_order).index(i) for i in mwanted_order]
        if (np.sum(mcolor_dict ==None)!=0) or (np.sum(mcolor_dict.index.isin(mwanted_order))!=len(mwanted_order)):
            mcolor_dict={}
            for item in mwanted_order:
                mcolor_dict[item]=random.sample(range(0, 255), 3)
        mcolor_dict = mcolor_dict.map(lambda x: list(map(lambda y: y/255., x)))
        #mnewcolors = array(list(mcolor_dict[mprotogruop].values))
        normalizerTest=0.9*mdf_train_set.values.max(1)[:,np.newaxis]
        normalizedValue=(mdf_train_set.sub(mdf_train_set.min(1),0).div(normalizerTest,0).fillna(0).values).T

        ####
        # must be np.float32
        ####
        dataRef=net.predict_proba((normalizedValue).astype(np.float32))[:,datarefplot.uns["Celltype_OrderNumber"]]
        mreordername=[]
        for i in mreorder_ix:
            mreordername.append(list(mclasses_names)[i])
        #ollist=[datarefplot.uns['Celltype_Score_RefCellType'][i] for i in datarefplot.uns['Celltype_OrderNumber']]
        #dfprobCL=pd.DataFrame(dataRef*100,   index=mdf_train_set.columns,columns= collist)
        dfprobCL = pd.DataFrame(dataRef * 100, index=mdf_train_set.columns,
                                columns=datarefplot.uns['Celltype_Score_RefCellType'])
        #dfnewcl=pd.DataFrame(array([xtest,ytest]).T, index=mdf_train_set.columns)
        datax.obsm["Celltype_Score"]=dfprobCL.values
        datax.uns["Celltype_Score_RefCellType"]=dfprobCL.columns.tolist()
        datax.uns["Celltype_OrderNumber"]=mreorder_ix
        if radarplot ==True:
            axm, dfclRef = RadarVisualization(refdataLR=dataRef,
                                              mreorder_ix=datarefplot.uns["Celltype_OrderNumber"],
                                            mtrain_index=datarefplot.obs["mtrain_index"].values,
                                              learninggroup="test",
                                              Ncolm=ncolnm, bbValue=bbValue,
                                              mdf_train_set=mdf_train_set,
                                              mprotogruop=datax.obs["Cluster"].values,
                                                    dataclpn=dataRef,
                                                    mwanted_order=mwanted_order,
                                              mclasses_names=datarefplot.uns["mclasses_names"],
                                                    mcolor_dict=mcolor_dict,
                                              fontsizeValue=int(100/int(len(mwanted_order)**0.5)))
            datax.obsm["CelltypeScoreCoordinates"]=dfclRef.values
        return datax



        return datax



def MVgene_Scaling(datax,score, commongene, mprotogruop,tftable,thrs,
                   std_scaling=False, TPTT=10000, sharedMVgenes=None,
                   learninggroup="train"):
    print("CamelRunning---GenesScaling......")
    dfpfc = pd.DataFrame(datax.X.T, index=datax.var.index, columns=datax.obs.index)
    dfpfcclus=pd.DataFrame([datax.obs["Cluster"].values.tolist(),datax.obs["Cluster"].values.tolist()],
                           index=["bk","Cluster"], columns=datax.obs.index)
    if learninggroup == "train":
        list_genes = datax.var.index[datax.var["MVgene"]].tolist()
        df_dev_rev = dfpfc.iloc[np.argsort(score)[::-1], :].iloc[max(min(-thrs, -2000), -5000):, :]
        tflist = pd.read_table(tftable, index_col=0, header=0, sep="\t").index.tolist()
        sharedMVgenes = list(set(list_genes + tflist) - set(df_dev_rev.index))
        if  np.sum(TPTT) != 0:
            dfpfc = (dfpfc / dfpfc.sum()).multiply(TPTT, axis=0).fillna(0)
        if std_scaling == True:
            scalepfc = dfpfc.div(dfpfc.std(1), axis=0).dropna(0)
        scalepfc = dfpfc.astype(float).dropna(0)
        scalepfc = dfpfc.div(dfpfc.std(1), axis=0)
        scalepfc = scalepfc.dropna(0)
        dfpfc_dev = scalepfc.loc[set(scalepfc.index) & set(sharedMVgenes)].dropna()
        dfpfc_dev_log = np.log2(dfpfc_dev + 1)
        dfpfc_dev_all = dfpfc_dev_log.T.join(dfpfcclus.T, how="inner").dropna()
        bool1 = mprotogruop != "nan"
        mclasses_names, mclasses_index = np.unique(mprotogruop[bool1], return_inverse=True, return_counts=False)
        mtrain_index = mclasses_index
        mdf_train_set = dfpfc_dev_log.loc[:, bool1].copy()
        # mdf_train_set = dfpfc_dev_log.copy()
        mdf_train_set = mdf_train_set.loc[mdf_train_set.sum(1) > 0]
        sharedMVgenes = mdf_train_set.index.tolist()
        #datax.uns["train_set"]=mdf_train_set
        datax.obsm["train_set_values"] = mdf_train_set.values.T
        datax.uns["train_set_gene"] = mdf_train_set.index.values
        datax.uns["mclasses_names"] = mclasses_names
        datax.obs["mtrain_index"] = mtrain_index
        refgenelist = np.in1d(datax.var.index, sharedMVgenes)
        datax.var["RefGeneList"]=refgenelist
        print("CamelRunning---TrainingGenesScaling......Finished")
        return datax
    elif learninggroup == "test":
        dfpfc = dfpfc.reindex(commongene).fillna(0).astype(float)
        if np.sum(TPTT) != 0:
            dfpfc = (dfpfc / dfpfc.sum()).multiply(TPTT, axis=0).fillna(0)

        if std_scaling == True:
            scalegbm = dfpfc.div(dfpfc.std(1), axis=0).dropna(0)
        else:
            scalegbm = dfpfc.astype(float).dropna(0)

        dfgbm_dev = scalegbm.reindex(sharedMVgenes).fillna(0)
        dfgbm_dev_log = np.log2(dfgbm_dev + 1).fillna(0)
        # df_dev_gbm = df_dev_gbm.loc[mdf_train_set.index].fillna(0)
        dfgbm_dev_all = dfgbm_dev_log.T.join(dfpfcclus.loc["Cluster"].T, how="inner").T
        # dfgbmcol = dfgbm_dev_all.iloc[-1:, :]
        # dfgbm = dfgbm_dev_all.iloc[:-1, :]
        dfclpncol = dfgbm_dev_all.iloc[-1:, :]
        dfclpn = dfgbm_dev_all.iloc[:-1, :]
        protogruop = dfclpncol.loc["Cluster"].values
        bool1 = protogruop != 'none'
        classes_names, classes_index = np.unique(protogruop[bool1], return_inverse=True, return_counts=False)
        train_index = classes_index
        # dfgbm_train_set = dfgbm_dev_log.loc[:, bool1].copy()
        # train_index = classes_index
        # dfgbm_train_set = dfgbm_train_set.loc[dfgbm_train_set.sum(1) > 0]
        dfclpn = dfclpn.loc[:, bool1].copy()
        df_train_setclpn = dfclpn.loc[dfclpn.sum(1) > 0]
        df_train_setclpn = df_train_setclpn.reindex(sharedMVgenes).fillna(0)
        datax.obsm["test_set_values"] =  df_train_setclpn.values.T
        datax.uns["train_set_gene"] =  df_train_setclpn.index.values
        datax.uns["mclasses_names"] = classes_names
        datax.obs["mtrain_index"] = classes_index
        testgenelist = np.in1d(datax.var.index, sharedMVgenes)
        datax.var["RefGeneList"] = testgenelist
        print("CamelRunning---TestGenesScaling......Finished")
        return datax
        #return df_train_setclpn, dfclpncol, protogruop



def NNclassifer(datax, epochNum=100, learningRate=0.05, verbose=0,
                optimizerMmentum=0.8,
                dropout=0.3):
    # nist_d=mdf_train_set.shape[0],
    # hidden_d = int(mnist_d / 5),
    # output_d = len(unique(mtrain_index))):
    # from skorch import NeuralNetClassifier
    # import Classifier3Layers
    # import LossTweet
    deviceSel = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("CamelRunning---NNclasffier_in_%s......."%deviceSel)
    mdf_train_set = pd.DataFrame(datax.obsm["train_set_values"].T, index=datax.uns["train_set_gene"],
                                 columns=datax.obs.index)
    mtrain_index= datax.obs["mtrain_index"]
    class Classifier3Layers(nn.Module):
        def __init__(
                self,
                input_dim=mdf_train_set.shape[0],
                hidden_dim=int(mdf_train_set.shape[0] / 5),
                output_dim=len(np.unique(mtrain_index)),
                dropout=0.3,
        ):
            super(Classifier3Layers, self).__init__()
            self.dropout = nn.Dropout(dropout)

            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=-1)
            return X


    net = NeuralNetClassifier(
        Classifier3Layers(
            input_dim=mdf_train_set.shape[0],
            hidden_dim=int(mdf_train_set.shape[0] / 5),
            output_dim=len(np.unique(mtrain_index)),
            dropout=0.5,
        ).float(),
        max_epochs=epochNum,
        lr=learningRate,
        verbose=0,
        optimizer__momentum=optimizerMmentum,
        module__dropout=dropout,
        #optimizer__nesterov=True,
        device=deviceSel,
        # callbacks=[acTweet(max_loss=0.2)]
    )

    normalizer = 0.9 * mdf_train_set.values.max(1)[:, np.newaxis]
    net.fit((mdf_train_set.values / normalizer).T.astype(np.float32), mtrain_index)
    # noticeMSG
    print("CamelRunning---NNclasffier_in_%s.......Finished" % deviceSel)
    return net




def AccuracyPlot( nnModel, accCutoff=0.95, Xlow=-1,Ylow=0.5, Yhigh=1,):
    fig_args = {'figsize': (6, 3), 'facecolor': 'white', 'edgecolor': 'white'}
    #acc = net.history[:, 'valid_acc'], accCutoff = 0.95,
    #Xlow = -1, Xhigh = len(nnModel.history[:, 'valid_acc']) + 1,
    acc = nnModel.history[:, 'valid_acc']
    Xhigh = len(nnModel.history[:, 'valid_acc']) + 1
    fig = plt.figure(**fig_args)
    ax = fig.add_subplot(111)
    ax.plot(np.array([abs(i) for i in range(Xhigh-1)]),np.array( acc ), c='k', lw=2 )

    ax.axhline( accCutoff, c='b' )
    #axvline( 35 , c='r')
    plt.ylabel('Accuracy Score', fontsize=15)
    plt.xlabel('Epoches', fontsize=15)
    plt.xlim( Xlow, Xhigh)
    plt.ylim(Ylow, Yhigh)
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax
