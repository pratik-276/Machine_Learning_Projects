import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

with open("train.csv", 'r') as file:
    reviews = list(csv.reader(file))

with open("test.csv", 'r') as file:
    test = list(csv.reader(file))

actual = [int(r[1]) for r in test]

vectorizer = CountVectorizer(stop_words='english', max_df=.05)
train_features = vectorizer.fit_transform([r[0] for r in reviews])
test_features = vectorizer.transform([r[0] for r in test])

nb = MultinomialNB()
nb.fit(train_features, [int(r[1]) for r in reviews])

predictions = nb.predict(test_features)

fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
print("Multinomal naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))