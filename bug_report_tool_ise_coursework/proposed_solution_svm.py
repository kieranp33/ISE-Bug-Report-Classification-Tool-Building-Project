import pandas as pd
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (GridSearchCV,StratifiedKFold) # Optimisating parameters
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.stats import ranksums

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()



# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'caffe' # Please specify the project name in order to run the baseline code to obtain the evaluation metric scores for this project
path = f'{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # consistent shuffling for multiple runs

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])
datafile = 'Title+Body.csv'

REPEAT = 10
out_csv_name = f'{project}_NB.csv'

data = pd.read_csv(datafile).fillna('')
text_col = 'text'

data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []

for repeated_time in range(REPEAT):
    # --- 4.1 Split into train/test ---
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]

    # --- 4.2 TF-IDF vectorization ---
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1100
    )

    X_train = tfidf.fit_transform(train_text).toarray()
    X_test = tfidf.transform(test_text).toarray()

    # Need to implement gridsearch for best parameter search
    # Need to find best value for gamma and regularisation,C
    param_grid = [
        {'C':[0.5,1,10,100],
        'gamma':['scale',1,0.1,0.01,0.001,0.0001],
        'kernel':['linear','rbf'],
         }
    ]

    k_fold=StratifiedKFold(n_splits=10,shuffle=False)

    optional_params=GridSearchCV(
        SVC(class_weight='balanced'),
        param_grid,
        cv=k_fold,  # This was changed to conduct stratifed sampling due to the imbalance in some systems - aimed to make a fairer comparison between baseline and proposed solution
        scoring='f1', # Using f1 for model evaluationss
        verbose=0
    )

    optional_params.fit(X_train,y_train)
    best_clf=optional_params.best_estimator_
    y_pred=best_clf.predict(X_test)




    print(f"Optimal Parameter Setting for Repeated Experiment Run {repeated_time}: {optional_params.best_params_}") # These optimal parameters for each run can be verfied in report (figure 2)
    # Below is the evaluation metric score for a given project. There are 10 values for each metric
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Repeated Experiment Run {repeated_time}: Accuracy:      {acc:.4f}")
    accuracies.append(acc)

    # Precision (macro)
    prec = precision_score(y_test, y_pred, average='macro')
    print(f"Repeated Experiment Run {repeated_time}: Precision:      {prec:.4f}")
    precisions.append(prec)

    # Recall (macro)
    rec = recall_score(y_test, y_pred, average='macro')
    print(f"Repeated Experiment Run {repeated_time}: Recall:      {rec:.4f}")
    recalls.append(rec)

    # F1 Score (macro)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Repeated Experiment Run {repeated_time}: F1_Score:    {f1:.4f}")
    f1_scores.append(f1)

    # AUC
    # If labels are 0/1 only, this works directly.
    # If labels are something else, adjust pos_label accordingly.
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    print(f"Repeated Experiment Run {repeated_time}: AUC:    {auc_val:.4f}")
    auc_values.append(auc_val)

    print("\n");





# --- 4.5 Aggregate results ---
final_accuracy  = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall    = np.mean(recalls)
final_f1        = np.mean(f1_scores)
final_auc       = np.mean(auc_values)


# Below are the baseline evaluation metrics that were produced in the baseline line code. This can be verfied by running the baseline code in the github repo and is reported as figure 8,9,10,11,12
baseline_precision_caffe=[0.5764635603345281,0.5817245817245817,0.5330882352941176,0.504778972520908,0.6296296296296297,0.5496894409937888,0.5535714285714286,0.5354037267080746,0.5440993788819876,0.5997474747474747]
baseline_recall_caffe=[0.7051282051282051,0.739622641509434,0.6018867924528302,0.5185185185185186,0.803921568627451,0.6282051282051282,0.575,0.6075471698113208,0.5994397759103641,0.7532051282051282]
baseline_f1_scores_caffe=[0.5129609346476817,0.5538461538461539,0.47126436781609193,0.40569259962049337,0.5839311334289814,0.41528455284552845,0.5538461538461539,0.3894736842105263,0.5054263565891473,0.5762987012987013]


baseline_precision_incubator_mxnet=[0.629585326953748,0.6757105943152455,0.6280303030303029,0.6952380952380952,0.66875,0.6563063063063063,0.6534820824881676,0.6522556390977443,0.6523809523809524,0.6182733255903987]
baseline_recall_incubator_mxnet=[0.7747252747252747,0.6931818181818181,0.7857142857142858,0.7603174603174603,0.7934782608695652,0.7599250936329588,0.7414893617021276,0.7571428571428571,0.6818181818181819,0.7582417582417582]
baseline_f1_scores_incubator_mxnet=[0.6233295866117148,0.6835699797160244,0.6012547926106657,0.7187288708586883,0.6976744186046512,0.6738922972051806,0.6807857581338244,0.6718301778542742,0.6640826873385013,0.5998075998075998]

baseline_precision_keras=[0.6724537037037037,0.7155782848151062,0.594718992248062,0.6558794466403162,0.6467348544453186,0.7000517598343685,0.7243589743589743,0.6865422123154081,0.6662921348314607,0.681712962962963]
baseline_recall_keras=[0.7578746971270336,0.8336038961038961,0.6714912280701755,0.7315596330275229,0.7053211009174312,0.7675666320526133,0.8002450980392157,0.7198686371100165,0.7522727272727272,0.764487870619946]
baseline_f1_scores_keras=[0.6726744569881824,0.740611691831204,0.5861764705882353,0.6646886394509187,0.6582658265826582,0.7169907508557279,0.7292929292929293,0.6984698469846984,0.6779549923530697,0.6837136113296617]

baseline_precision_tensorflow=[0.6388699020277968,0.6231884057971014,0.6273286140089419,0.6193438140806562,0.664878612716763,0.6404420141262246,0.6512667660208644,0.6261335156071999,0.6782967032967033,0.6463326071169209]
baseline_recall_tensorflow=[0.7280209502431725,0.696114146933819,0.7278333333333333,0.71825,0.7667789001122334,0.7485483870967742,0.7168803418803419,0.7306666666666667,0.7834680061148722,0.7642406360134415]
baseline_f1_scores_tensorflow=[0.6192636629119733,0.6187138406764575,0.6132007732670534,0.5913732277174999,0.6577457264957265,0.6175392635520449,0.6483084185680567,0.5993077281501697,0.6688152922871748,0.6198329716577892]


baseline_precision_pytorch=[0.6483180428134556,0.6374671916010499,0.6267184035476718,0.6118527508090615,0.6181578947368421,0.5773001508295625,0.6142399267399268,0.5881307746979388,0.5951210951210951,0.6071428571428571]
baseline_recall_pytorch=[0.7836257309941521,0.6216608594657376,0.7181297709923664,0.7025641025641025,0.7209645669291338,0.6714046822742474,0.7190517998244074,0.6977671451355663,0.7263993316624895,0.7977941176470589]
baseline_f1_scores_pytorch=[0.6672176308539945,0.6283076923076923,0.6405006462145432,0.6149157181066119,0.5664356435643564,0.5842302878598248,0.6273228803716608,0.5447889750215331,0.531055900621118,0.5497424776362158]





print("=== SVM + TF-IDF Results ===")
print(f"Number of repeats:     {REPEAT}") # Prints the average baseline metric values for the 5 metrics used in a specified project. Can be verfied with results shown in report (figures 3,4,5,6,7)
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

print("\n")

stat_test_precision = 0;
stat_test_recall = 0;
stat_test_f1 =0;
# Below is the Wilcoxon Rank Sum Tests for conducting statistical test between the baseline and proposed solution for 3 evaluation metrics
print("=== Wilcoxon Rank Sum Test Results ===")
if project == "caffe":

    stat_test_precision=ranksums(baseline_precision_caffe,precisions).pvalue
    stat_test_recall=ranksums(baseline_recall_caffe,recalls).pvalue
    stat_test_f1=ranksums(baseline_f1_scores_caffe,f1_scores).pvalue

    print(f"Wilcoxon Rank Sum P-Value for Precision {stat_test_precision}")
    print(f"Wilcoxon Rank Sum P-Value for Recall {stat_test_recall}")
    print(f"Wilcoxon Rank Sum P-Value for F1-Score {stat_test_f1}")

if project == "incubator-mxnet":

    stat_test_precision=ranksums(baseline_precision_incubator_mxnet,precisions).pvalue
    stat_test_recall=ranksums(baseline_recall_incubator_mxnet,recalls).pvalue
    stat_test_f1=ranksums(baseline_f1_scores_incubator_mxnet,f1_scores).pvalue

    print(f"Wilcoxon Rank Sum P-Value for Precision {stat_test_precision}")
    print(f"Wilcoxon Rank Sum P-Value for Recall {stat_test_recall}")
    print(f"Wilcoxon Rank Sum P-Value for F1-Score {stat_test_f1}")

if project == "keras":

    stat_test_precision=ranksums(baseline_precision_keras,precisions).pvalue
    stat_test_recall=ranksums(baseline_recall_keras,recalls).pvalue
    stat_test_f1=ranksums(baseline_f1_scores_keras,f1_scores).pvalue

    print(f"Wilcoxon Rank Sum P-Value for Precision {stat_test_precision}")
    print(f"Wilcoxon Rank Sum P-Value for Recall {stat_test_recall}")
    print(f"Wilcoxon Rank Sum P-Value for F1-Score {stat_test_f1}")

if project == "tensorflow":

    stat_test_precision=ranksums(baseline_precision_tensorflow,precisions).pvalue
    stat_test_recall=ranksums(baseline_recall_tensorflow,recalls).pvalue
    stat_test_f1=ranksums(baseline_f1_scores_tensorflow,f1_scores).pvalue

    print(f"Wilcoxon Rank Sum P-Value for Precision {stat_test_precision}")
    print(f"Wilcoxon Rank Sum P-Value for Recall {stat_test_recall}")
    print(f"Wilcoxon Rank Sum P-Value for F1-Score {stat_test_f1}")

if project == "pytorch":

    stat_test_precision=ranksums(baseline_precision_pytorch,precisions).pvalue
    stat_test_recall=ranksums(baseline_recall_pytorch,recalls).pvalue
    stat_test_f1=ranksums(baseline_f1_scores_pytorch,f1_scores).pvalue

    print(f"Wilcoxon Rank Sum P-Value for Precision {stat_test_precision}")
    print(f"Wilcoxon Rank Sum P-Value for Recall {stat_test_recall}")
    print(f"Wilcoxon Rank Sum P-Value for F1-Score {stat_test_f1}")


print('\n')
print("=== Standard Deviation Results ===") # Prints the standard deviation values for the 5 metrics used in a specified project. Can be verified with results shown in report (figures 3,4,5,6,7)
print(f"Standard Deviation for Accuracy:      {np.std(accuracies):.4f}")
print(f"Standard Deviation for Precision:     {np.std(precisions):.4f}")
print(f"Standard Deviation for Recall:        {np.std(recalls):.4f}")
print(f"Standard Deviation for F1 score:      {np.std(f1_scores):.4f}")
print(f"Standard Deviation for AUC:           {np.std(auc_values):.4f}")
