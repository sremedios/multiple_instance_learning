def get_CI(y_true, y_pred):
    n_bootstraps = 100
    rng_seed = 100 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        fpr, tpr, thresholds = roc_curve(y_true[indices], y_pred[indices])
        #fprs[i] = fpr
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    return mean_auc, std_auc, mean_fpr, mean_tpr, tprs_lower, tprs_upper

from scipy import interp
df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/for_spore/norm_all.csv')

print (len(df.loc[df['source'] == 'nlst']))


tar = df['gt']#.tolist() 
plco_pred = df['plco_risk']#.tolist()
kaggle_pred = df['kaggle_risk']#.tolist()
drnn_pred = df['drnn_pred']#.tolist()
#print (sum(tar))

df_comb = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/for_spore/10fold_nomcl/svm_linear_val/combine.csv')
tar_comb = df_comb['gt']
pred_comb = df_comb['pred']

df_log = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/for_spore/10fold_nomcl/val_result/clinical_log/combine.csv')

tar_log = df_log['gt']
pred_log = df_log['pred']

#print("Original ROC area: {:0.3f}".format(roc_auc_score(y_true, y_pred)))




log_mean_auc, log_std_auc, log_mean_fpr, log_mean_tpr, log_tprs_lower, log_tprs_upper = get_CI(tar_log, pred_log)
plco_mean_auc, plco_std_auc,plco_mean_fpr, plco_mean_tpr, plco_tprs_lower, plco_tprs_upper = get_CI(tar, plco_pred)
k_mean_auc, k_std_auc,k_mean_fpr, k_mean_tpr, k_tprs_lower, k_tprs_upper = get_CI(tar, kaggle_pred)
drnn_mean_auc, drnn_std_auc,drnn_mean_fpr, drnn_mean_tpr, drnn_tprs_lower, drnn_tprs_upper = get_CI(tar, drnn_pred)
comb_mean_auc, comb_std_auc,comb_mean_fpr, comb_mean_tpr, comb_tprs_lower, comb_tprs_upper = get_CI(tar_comb, pred_comb)


plt.plot(plco_mean_fpr, plco_mean_tpr, color='b',
          label=r'PLCO 2012: %0.2f $\pm$ %0.2f' % (plco_mean_auc, plco_std_auc),
         lw=2, alpha=.8)
plt.fill_between(plco_mean_fpr, plco_tprs_lower, plco_tprs_upper, color='b', alpha=.2)

plt.plot(k_mean_fpr, k_mean_tpr, color='deeppink',
          label=r'Kaggle Winner: %0.2f $\pm$ %0.2f' % (k_mean_auc, k_std_auc),
         lw=2, alpha=.8)
plt.fill_between(k_mean_fpr, k_tprs_lower, k_tprs_upper, color='deeppink', alpha=.2)

plt.plot(drnn_mean_fpr, drnn_mean_tpr, color='aqua',
          label=r'DLSTM: %0.2f $\pm$ %0.2f' % (drnn_mean_auc, drnn_std_auc),
         lw=2, alpha=.8)
plt.fill_between(drnn_mean_fpr, drnn_tprs_lower, drnn_tprs_upper, color='aqua', alpha=.2)

plt.plot(log_mean_fpr, log_mean_tpr, color='darkorange',
          label=r'LICDC(logistic regression): %0.2f $\pm$ %0.2f' % (log_mean_auc, log_std_auc),
         lw=2, alpha=.8)
plt.fill_between(log_mean_fpr, log_tprs_lower, log_tprs_upper, color='darkorange', alpha=.2)

plt.plot(comb_mean_fpr, comb_mean_tpr, color='cornflowerblue',
          label=r'LICDC(SVM): %0.2f $\pm$ %0.2f' % (comb_mean_auc, comb_std_auc),
         lw=2, alpha=.8)
plt.fill_between(comb_mean_fpr, comb_tprs_lower, comb_tprs_upper, color='cornflowerblue', alpha=.2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#
plt.savefig('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/for_spore/for_spore_result_CI.eps')
plt.show()

