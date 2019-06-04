import re
from string import punctuation
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


df = pd.read_csv('data\quora_features.csv', delimiter=",")
SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

def clean(text, stem_words=True):

    def pad_str(s):
        return ' ' + s + ' '

    if pd.isnull(text):
        return ''

    if type(text) != str or text == '':
        return ''

    # Cleaning the text
    text = re.sub("\'s", " ",
                  text)  # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)

    # removing comma between the numbers, i.e. 10,000 -> 10000

    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)


    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']),
                  text)


    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)

    # cleaning text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE)
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)


    # Remove punctuation marks from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
    # Return a list of words
    return text


df['question1'] = df['question1'].apply(clean)
df['question2'] = df['question2'].apply(clean)

#------------------ nGRAM CHAR TFIDF ---------------------

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(pd.concat((df['question1'],df['question2'])).unique())
trainq1_trans = tfidf_vect_ngram_chars.transform(df['question1'].values)
trainq2_trans = tfidf_vect_ngram_chars.transform(df['question2'].values)
labels = df['is_duplicate'].values
X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

# --------------- XGBOOSTING ----------------------
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_prediction = xgb_model.predict(X_valid)

# ---------------- RANDOM FOREST -----------------------
from sklearn.ensemble import RandomForestClassifier
randomModel = RandomForestClassifier()
randomModel.fit(X_train, y_train)
random_pred = randomModel.predict(X_valid)


modelSVM = SVC(kernel='linear', C = 0.1)
modelSVM.fit(X_train, y_train)
pred = modelSVM.predict(X_valid)



print("Accuracy Score XGBoost TFIDF Char NGrams: ",accuracy_score(y_valid, xgb_prediction))
print(confusion_matrix(y_valid, xgb_prediction))
print("ROC-AUC XGBoost TFIDF Char NGrams: ",roc_auc_score(xgb_prediction, y_valid))

print("Accuracy Score Random Forest TFIDF Char NGrams: ",accuracy_score(y_valid, random_pred))
print("ROC-AUC RF TFIDF Char NGrams: ",roc_auc_score(random_pred, y_valid))
print(confusion_matrix(y_valid, random_pred))

print("Accuracy Score SVM: ",accuracy_score(y_valid, pred))
print("ROC-AUC SVM TFIDF Char NGrams: ",roc_auc_score(pred, y_valid))
print(confusion_matrix(y_valid, pred))

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
lr_pred = clf.predict(X_valid)
print("Accuracy Score Logistic Regression TFIDF Ngrams Char: ", accuracy_score(y_valid, lr_pred))
print("ROC-AUC Logistic Regression TFIDF Ngrams Char: ", roc_auc_score(lr_pred, y_valid))
print(confusion_matrix(y_valid, lr_pred))


# ---------------- COUNT VECTORIZER --------------

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(pd.concat((df['question1'],df['question2'])).unique())
trainq1_trans = count_vect.transform(df['question1'].values)
trainq2_trans = count_vect.transform(df['question2'].values)
labels = df['is_duplicate'].values
X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

# --------------- XGBOOSTING ----------------------
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_prediction = xgb_model.predict(X_valid)

# ---------------- RANDOM FOREST -----------------------

randomModel = RandomForestClassifier()
randomModel.fit(X_train, y_train)
random_pred = randomModel.predict(X_valid)


modelSVM = SVC(kernel='linear', C = 0.1)
modelSVM.fit(X_train, y_train)
pred = modelSVM.predict(X_valid)



print("Accuracy Score XGBoost Count Vectorizer: ",accuracy_score(y_valid, xgb_prediction))
print(confusion_matrix(y_valid, xgb_prediction))
print("ROC-AUC XGB Count Vectorizer: ",roc_auc_score(xgb_prediction, y_valid))

print("Accuracy Score Random Forest Count Vectorizer: ",accuracy_score(y_valid, random_pred))
print("ROC-AUC RF Count Vectorizer: ",roc_auc_score(random_pred, y_valid))
print(confusion_matrix(y_valid, random_pred))

print("Accuracy Score SVM Count Vectorizer: ",accuracy_score(y_valid, pred))
print("ROC-AUC SVM Count Vectorizer: ",roc_auc_score(pred, y_valid))
print(confusion_matrix(y_valid, pred))

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
lr_pred = clf.predict(X_valid)
print("Accuracy Score Logistic Regression Count Vectorizer: ", accuracy_score(y_valid, lr_pred))
print("ROC-AUC Logistic Regression Count Vectorizer: ", roc_auc_score(lr_pred, y_valid))
print(confusion_matrix(y_valid, lr_pred))


# -------------- TFIDF NGRAM WORD ---------------------

tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=(2,3), token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(pd.concat((df['question1'],df['question2'])).unique())
trainq1_trans = tfidf_vect.transform(df['question1'].values)
trainq2_trans = tfidf_vect.transform(df['question2'].values)
labels = df['is_duplicate'].values
X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

# --------------- XGBOOSTING ----------------------
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_prediction = xgb_model.predict(X_valid)

# ---------------- RANDOM FOREST -----------------------

randomModel = RandomForestClassifier()
randomModel.fit(X_train, y_train)
random_pred = randomModel.predict(X_valid)


modelSVM = SVC(kernel='linear', C = 0.5)
modelSVM.fit(X_train, y_train)
pred = modelSVM.predict(X_valid)



print("Accuracy Score XGBoost TFDIF NGRAM WORD: ",accuracy_score(y_valid, xgb_prediction))
print(confusion_matrix(y_valid, xgb_prediction))
print("ROC-AUC XGBoost TFDIF NGRAM WORD: ",roc_auc_score(xgb_prediction, y_valid))

print("Accuracy Score Random Forest TFDIF NGRAM WORD: ",accuracy_score(y_valid, random_pred))
print("ROC-AUC RF TFDIF NGRAM WORD: ",roc_auc_score(random_pred, y_valid))
print(confusion_matrix(y_valid, random_pred))

print("Accuracy Score SVM TFDIF NGRAM WORD: ",accuracy_score(y_valid, pred))
print("ROC-AUC SVM TFDIF NGRAM WORD: ",roc_auc_score(pred, y_valid))
print(confusion_matrix(y_valid, pred))

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
lr_pred = clf.predict(X_valid)
print("Accuracy Score Logistic Regression TFDIF NGRAM WORD: ", accuracy_score(y_valid, lr_pred))
print("ROC-AUC Logistic Regression TFDIF NGRAM WORD: ", roc_auc_score(lr_pred, y_valid))
print(confusion_matrix(y_valid, lr_pred))


def replace_missing_value(df, number_features):

    imputer = Imputer(strategy="median")
    df_num = df[number_features]
    imputer.fit(df_num)
    X = imputer.transform(df_num)
    res_def = pd.DataFrame(X, columns=df_num.columns)
    return res_def

df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)
#------------- LENGTH FEATURES --------------------------

col = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2', 'common_words']
X = df[col]
Y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,Y, test_size = 0.33, random_state = 42)

# --------------- XGBOOSTING ----------------------
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_prediction = xgb_model.predict(X_valid)

# ---------------- RANDOM FOREST -----------------------

randomModel = RandomForestClassifier()
randomModel.fit(X_train, y_train)
random_pred = randomModel.predict(X_valid)


modelSVM = SVC(kernel='linear', C = 0.1)
modelSVM.fit(X_train, y_train)
pred = modelSVM.predict(X_valid)



print("Accuracy Score XGBoost Length Features: ",accuracy_score(y_valid, xgb_prediction))
print(confusion_matrix(y_valid, xgb_prediction))
print("ROC-AUC XGBoost Length Features: ",roc_auc_score(xgb_prediction, y_valid))

print("Accuracy Score Random Forest Length Features: ",accuracy_score(y_valid, random_pred))
print("ROC-AUC RF Length Features: ",roc_auc_score(random_pred, y_valid))
print(confusion_matrix(y_valid, random_pred))

print("Accuracy Score SVM Length Features: ",accuracy_score(y_valid, pred))
print("ROC-AUC SVM Length Features: ",roc_auc_score(pred, y_valid))
print(confusion_matrix(y_valid, pred))

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
lr_pred = clf.predict(X_valid)
print("Accuracy Score Logistic Regression Length Features: ", accuracy_score(y_valid, lr_pred))
print("ROC-AUC Logistic Regression Length Features: ", roc_auc_score(lr_pred, y_valid))
print(confusion_matrix(y_valid, lr_pred))


#-------------------------- FUZZY FEATURES ----------------

col = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
X = df[col]
Y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

# --------------- XGBOOSTING ----------------------
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_prediction = xgb_model.predict(X_valid)

# ---------------- RANDOM FOREST -----------------------

randomModel = RandomForestClassifier()
randomModel.fit(X_train, y_train)
random_pred = randomModel.predict(X_valid)


modelSVM = SVC(kernel='linear', C = 0.1)
modelSVM.fit(X_train, y_train)
pred = modelSVM.predict(X_valid)



print("Accuracy Score XGBoost Fuzzy Features: ",accuracy_score(y_valid, xgb_prediction))
print(confusion_matrix(y_valid, xgb_prediction))
print("ROC-AUC XGBoost Fuzzy Features: ",roc_auc_score(xgb_prediction, y_valid))

print("Accuracy Score Random Forest Fuzzy Features: ",accuracy_score(y_valid, random_pred))
print("ROC-AUC RF Fuzzy Features: ",roc_auc_score(random_pred, y_valid))
print(confusion_matrix(y_valid, random_pred))

print("Accuracy Score SVM Fuzzy Features: ",accuracy_score(y_valid, pred))
print("ROC-AUC SVM Fuzzy Features: ",roc_auc_score(pred, y_valid))
print(confusion_matrix(y_valid, pred))

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
lr_pred = clf.predict(X_valid)
print("Accuracy Score Logistic Regression Fuzzy Features: ", accuracy_score(y_valid, lr_pred))
print("ROC-AUC Logistic Regression Fuzzy Features: ", roc_auc_score(lr_pred, y_valid))
print(confusion_matrix(y_valid, lr_pred))


#--------------------- VECTOR DISTANCES ----------------

col = ['wmd', 'norm_wmd', 'cosine_distance', 'cityblock_distance','jaccard_distance', 'canberra_distance','euclidean_distance', 'minkowski_distance', 'braycurtis_distance',  'skew_q1vec','skew_q2vec', 'kur_q1vec','kur_q2vec' ]
res_df = replace_missing_value(df, col)
X = res_df[col]
Y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,Y, test_size = 0.33, random_state = 42)

X_valid.fillna(X_train.mean(), inplace=True)

# --------------- XGBOOSTING ----------------------
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_prediction = xgb_model.predict(X_valid)

# ---------------- RANDOM FOREST -----------------------

randomModel = RandomForestClassifier()
randomModel.fit(X_train, y_train)
random_pred = randomModel.predict(X_valid)


modelSVM = SVC(kernel='linear', C = 0.1)
modelSVM.fit(X_train, y_train)
pred = modelSVM.predict(X_valid)



print("Accuracy Score XGBoost Vector Distance Features: ",accuracy_score(y_valid, xgb_prediction))
print(confusion_matrix(y_valid, xgb_prediction))
print("ROC-AUC XGBoost Vector Distance Features: ",roc_auc_score(xgb_prediction, y_valid))

print("Accuracy Score Random Forest Vector Distance Features: ",accuracy_score(y_valid, random_pred))
print("ROC-AUC RF Vector Distance Features: ",roc_auc_score(random_pred, y_valid))
print(confusion_matrix(y_valid, random_pred))

print("Accuracy Score SVM Vector Distance Features: ",accuracy_score(y_valid, pred))
print("ROC-AUC SVM Vector Distance Features: ",roc_auc_score(pred, y_valid))
print(confusion_matrix(y_valid, pred))

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
lr_pred = clf.predict(X_valid)
print("Accuracy Score Logistic Regression Vector Distance Features: ", accuracy_score(y_valid, lr_pred))
print("ROC-AUC Logistic Regression Vector Distance Features: ", roc_auc_score(lr_pred, y_valid))
print(confusion_matrix(y_valid, lr_pred))


#----------------------- ALL FESTURES -------------------------

col = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2', 'common_words', 'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio','wmd', 'norm_wmd', 'cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec', 'kur_q2vec']
#X = df[col]
res_df = replace_missing_value(df, col)
X = res_df[col]
Y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,Y, test_size = 0.33, random_state = 42)

# --------------- XGBOOSTING ----------------------
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_prediction = xgb_model.predict(X_valid)

# ---------------- RANDOM FOREST -----------------------
randomModel = RandomForestClassifier()
randomModel.fit(X_train, y_train)
random_pred = randomModel.predict(X_valid)

modelSVM = SVC()
modelSVM.fit(X_train, y_train)
pred = modelSVM.predict(X_valid)



print("Accuracy Score XGBoost All Features: ",accuracy_score(y_valid, xgb_prediction))
print(confusion_matrix(y_valid, xgb_prediction))
print("ROC-AUC XGBoost All Features: ",roc_auc_score(xgb_prediction, y_valid))

print("Accuracy Score Random Forest All Features: ",accuracy_score(y_valid, random_pred))
print("ROC-AUC RF All Features: ",roc_auc_score(random_pred, y_valid))
print(confusion_matrix(y_valid, random_pred))

print("Accuracy Score SVM All Features: ",accuracy_score(y_valid, pred))
print("ROC-AUC SVM All Features: ",roc_auc_score(pred, y_valid))
print(confusion_matrix(y_valid, pred))

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
lr_pred = clf.predict(X_valid)
print("Accuracy Score Logistic Regression All Features: ", accuracy_score(y_valid, lr_pred))
print("ROC-AUC Logistic Regression All Features: ", roc_auc_score(lr_pred, y_valid))
print(confusion_matrix(y_valid, lr_pred))