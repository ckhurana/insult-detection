# coding: utf-8

"""
    Title: Insult Detection Model
    Subject: Natural Language Processing
    Authors:
        - Chirag Khurana
        - Pallavi S. Rawat
        - Shubham Goyal
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np
import spacy

# Unprocessed Data
full_df = pd.read_csv('../data/train.csv')
verify_df = pd.read_csv('../data/impermium_verification_labels.csv')
data = full_df.append(verify_df)
data.Comment = [x[1: -1] for x in data.Comment]

# Processed Data
full_df = pd.read_csv('../data/processed/train.csv')
verify_df = pd.read_csv('../data/processed/impermium_verification_labels.csv')
pdata = full_df.append(verify_df)

nlp = spacy.load('en')


def sanitize_wo_stopwords(sentence):    # Sanitize sentences to remove stop words
    doc = nlp(sentence)
    s = []
    for token in doc:
        if str(token.pos_) != 'SPACE' and not token.is_stop:
            s.append(token.text)
    return ' '.join(s)


# pdata.Comment = [sanitize_wo_stopwords(x[1: -1]) for x in pdata.Comment]
pdata.Comment = [x[1: -1] for x in pdata.Comment]   # Removing double quotes from start and end


# Balancing train data to perform better training
pdata_ni = pdata.query('Insult == 0')
pdata_i = pdata.query('Insult == 1')


ptrain_ni, ptest_ni = train_test_split(pdata_ni, test_size=0.6)

ptrain_i, ptest_i = train_test_split(pdata_i, test_size=0.2)


# Splitting Dataset
ptrain = ptrain_i.append(ptrain_ni)
ptest = ptest_i.append(ptest_ni)


"""
    Feature Extraction
"""

# TFIDF as feature
tfidf_w = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', use_idf=False, max_features=50000) 
tfidf_c = TfidfVectorizer(ngram_range=(3, 10), analyzer='char', use_idf=False, max_features=100000)

ptrain_data_w = tfidf_w.fit_transform(ptrain.Comment)
ptrain_data_c = tfidf_c.fit_transform(ptrain.Comment)


"""
    Classification of Insult
"""
# Helper Functions
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):   # Source: not our team
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


"""
    Multinomial Naive Bayes
    with Accuracy, Confusion Matrices
"""

print('Starting MultinomialNB Classifier')
insult_nb_w = MultinomialNB(alpha=0.01)
insult_nb_w.fit(ptrain_data_w, ptrain.Insult)

insult_nb_c = MultinomialNB(alpha=0.01)
insult_nb_c.fit(ptrain_data_c, ptrain.Insult)

ptest_data_w = tfidf_w.transform(ptest.Comment)
ptest_data_c = tfidf_c.transform(ptest.Comment)

predicted_nb_w = insult_nb_w.predict(ptest_data_w)
predicted_nb_c = insult_nb_c.predict(ptest_data_c)
print('Naive Bayes (Word Gram):', np.mean(predicted_nb_w == ptest.Insult))
print('Naive Bayes (Character Gram):', np.mean(predicted_nb_c == ptest.Insult))

predicted_nb_w_prob = insult_nb_w.predict_proba(ptest_data_w)
predicted_nb_c_prob = insult_nb_c.predict_proba(ptest_data_c)


# #### Confusion Matrix for NB Classifier on Word - N-grams
cnf_matrix_w = confusion_matrix(ptest.Insult, predicted_nb_w)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
print('Naive Bayes (Word): ', end='')
plot_confusion_matrix(cnf_matrix_w, classes=['Not Insult', 'Insult'],
                      title='NB - Normalized confusion matrix')


# Confusion Matrix for NB Classifier on Character - N-grams
cnf_matrix_c = confusion_matrix(ptest.Insult, predicted_nb_c)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
print('Naive Bayes (Character): ', end='')
plot_confusion_matrix(cnf_matrix_c, classes=['Not Insult', 'Insult'],
                      title='NB - Normalized confusion matrix')
plt.show()

# ROC Curve and AUC for NB Classifier on Word- N-grams
print('MultinomialNB (Words Gram): ROC Curve')
print('MultinomialNB AUC (Word Gram):', roc_auc_score(ptest.Insult, predicted_nb_w_prob[:, 1:]))
fpr, tpr, _ = roc_curve(ptest.Insult, predicted_nb_w_prob[:, 1:])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('NB - ROC curve for insult classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

# ROC Curve and AUC for NB Classifier on Character- N-grams
print('MultinomialNB (Character Gram): ROC Curve')
print('MultinomialNB AUC (Character Gram):', roc_auc_score(ptest.Insult, predicted_nb_c_prob[:, 1:]))
fpr, tpr, _ = roc_curve(ptest.Insult, predicted_nb_c_prob[:, 1:])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('NB - ROC curve for insult classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()


"""
    LinearSVC
    with Accuracy, Confusion Matrices
"""

print('Starting SVM Classifier')

insult_svm_w = LinearSVC()
insult_svm_w.fit(ptrain_data_w, ptrain.Insult)

insult_svm_c = LinearSVC()
insult_svm_c.fit(ptrain_data_c, ptrain.Insult)

ptest_data_w = tfidf_w.transform(ptest.Comment)
ptest_data_c = tfidf_c.transform(ptest.Comment)

predicted_svm_w = insult_svm_w.predict(ptest_data_w)
predicted_svm_c = insult_svm_c.predict(ptest_data_c)
print('LinearSVC (SVM) (Word Gram):', np.mean(predicted_svm_w == ptest.Insult))
print('LinearSVC (SVM) (Character Gram):', np.mean(predicted_svm_c == ptest.Insult))


# Confusion Matrix for SVM Classifier on Word - N-grams
cnf_matrix_w = confusion_matrix(ptest.Insult, predicted_svm_w)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
print('LinearSVC (SVM) (Word Gram): ', end='')
plot_confusion_matrix(cnf_matrix_w, classes=['Not Insult', 'Insult'],
                      title='SVM - Normalized confusion matrix')


# Confusion Matrix for SVM Classifier on Character - N-grams
cnf_matrix_c = confusion_matrix(ptest.Insult, predicted_svm_c)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
print('LinearSVC (SVM) (Character Gram): ', end='')
plot_confusion_matrix(cnf_matrix_c, classes=['Not Insult', 'Insult'],
                      title='SVM - Normalized confusion matrix')
plt.show()


"""
    Logistic Regression
    with Accuracy, Confusion Matrices, ROC Curves, AUC scores
"""

print('Starting Logistic Regression')

insult_lr_w = LogisticRegression()
insult_lr_w.fit(ptrain_data_w, ptrain.Insult)

insult_lr_c = LogisticRegression()
insult_lr_c.fit(ptrain_data_c, ptrain.Insult)

ptest_data_w = tfidf_w.transform(ptest.Comment)
ptest_data_c = tfidf_c.transform(ptest.Comment)

predicted_lr_w = insult_lr_w.predict(ptest_data_w)
predicted_lr_c = insult_lr_c.predict(ptest_data_c)
print('Logistic Regression (Word Gram):', np.mean(predicted_lr_w == ptest.Insult))
print('Logistic Regression (Character Gram):', np.mean(predicted_lr_c == ptest.Insult))

predicted_lr_w_prob = insult_lr_w.predict_proba(ptest_data_w)
predicted_lr_c_prob = insult_lr_c.predict_proba(ptest_data_c)


# Confusion Matrix for Logistic Regression Classifier on Word - N-grams
cnf_matrix_w = confusion_matrix(ptest.Insult, predicted_lr_w)
np.set_printoptions(precision=2)
plt.figure()
print('Logistic Regression (Word Gram): ', end='')
plot_confusion_matrix(cnf_matrix_w, classes=['Not Insult', 'Insult'],
                      title='LR - Normalized confusion matrix')


# Confusion Matrix for Logistic Regression Classifier on Character - N-grams
cnf_matrix_c = confusion_matrix(ptest.Insult, predicted_lr_c)
np.set_printoptions(precision=2)
plt.figure()
print('Logistic Regression (Character Gram): ', end='')
plot_confusion_matrix(cnf_matrix_c, classes=['Not Insult', 'Insult'],
                      title='LR - Normalized confusion matrix')
plt.show()


# ROC Curve and AUC for Logistic Regression Classifier on Word - N-grams
print('Logistic Regression (Word Gram): ROC Curve')
print('LogisticRegression AUC (Word Gram):', roc_auc_score(ptest.Insult, predicted_lr_w_prob[:, 1:]))

fpr, tpr, _ = roc_curve(ptest.Insult, predicted_lr_w_prob[:, 1:])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('LR - ROC curve for insult classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()


# ROC Curve and AUC for Logistic Regression Classifier on Character - N-grams
print('Logistic Regression (Character Gram): ROC Curve')
print('LogisticRegression AUC (Character Gram):', roc_auc_score(ptest.Insult, predicted_lr_c_prob[:, 1:]))

fpr, tpr, _ = roc_curve(ptest.Insult, predicted_lr_c_prob[:, 1:])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('LR - ROC curve for insult classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
