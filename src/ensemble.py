from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np
import spacy


# #### Helper Functions

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
def sanitize_wo_stopwords(sentence):
    doc = nlp(sentence)
    s = []
    for token in doc:
        if str(token.pos_) != 'SPACE' and not token.is_stop:
            s.append(token.text)
    return ' '.join(s)


# pdata.Comment = [sanitize_wo_stopwords(x[1: -1]) for x in pdata.Comment]
pdata.Comment = [x[1: -1] for x in pdata.Comment]


pdata_ni = pdata.query('Insult == 0')
pdata_i = pdata.query('Insult == 1')


ptrain_ni, ptest_ni = train_test_split(pdata_ni, test_size=0.6)
# print(ptrain_ni.shape, ptest_ni.shape)

ptrain_i, ptest_i = train_test_split(pdata_i, test_size=0.2)
# print(ptrain_i.shape, ptest_i.shape)


# ### Splitting Dataset
ptrain = ptrain_i.append(ptrain_ni)
ptest = ptest_i.append(ptest_ni)


# ## Ensemble

from sklearn.pipeline import Pipeline
pipe_svm_w = Pipeline([
        ('tfidf_w', TfidfVectorizer(ngram_range=(1,3), use_idf=False, analyzer='word', max_features=5000)),
        ('svm_w', LinearSVC()),
    ])
pipe_svm_c = Pipeline([
        ('tfidf_c', TfidfVectorizer(ngram_range=(3,10), use_idf=False, analyzer='char', max_features=1000)),
        ('svm_c', LinearSVC()),
    ])

pipe_lr_w = Pipeline([
        ('tfidf_w', TfidfVectorizer(ngram_range=(1,3), use_idf=False, analyzer='word', max_features=5000)),
        ('lr_w', LogisticRegression()),
    ])
pipe_lr_c = Pipeline([
        ('tfidf_c', TfidfVectorizer(ngram_range=(3,10), use_idf=False, analyzer='char', max_features=1000)),
        ('lr_c', LogisticRegression()),
    ])

pipe_svm_w.fit(ptrain.Comment, ptrain.Insult)
pipe_svm_c.fit(ptrain.Comment, ptrain.Insult)
pipe_lr_w.fit(ptrain.Comment, ptrain.Insult)
pipe_lr_c.fit(ptrain.Comment, ptrain.Insult)
pipe_svm_w.score(ptest.Comment, ptest.Insult), pipe_lr_c.score(ptest.Comment, ptest.Insult), pipe_lr_c.score(ptest.Comment, ptest.Insult)

vote_clf = VotingClassifier(
    estimators=[
        ('svm_w', pipe_svm_w),
        # ('svm_c', pipe_svm_c),
        ('lr_w', pipe_lr_w),
        # ('lr_c', pipe_lr_c),
    ],
    # voting='soft'
)

vote_clf.fit(ptrain.Comment, ptrain.Insult)
pred = vote_clf.predict(ptest.Comment)
print('Ensemble of SVM, LogisticRegression:', np.mean(pred == ptest.Insult))
# pred_prob = vote_clf.predict_proba(ptest.Comment)
#
# fpr, tpr, thresholds = roc_curve(ptest.Insult, pred_prob[:, 1:])
# plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.title('ROC curve for insult classifier')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.grid(True)
# plt.show()
#

# Compute confusion matrix
cnf_matrix_vote = confusion_matrix(ptest.Insult, pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix_vote, classes=['Not Insult', 'Insult'], normalize=True,
                      title='Normalized confusion matrix')
plt.show()