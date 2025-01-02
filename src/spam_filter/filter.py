import pandas as pd
import numpy as np
import os
import email
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


def extract_email(file_path):
    """
    Read each email in the directory & parse email info
    Path: Variable used to define folder location
    """
    with open(file_path, 'rb') as f:
        msg = email.message_from_bytes(f.read())
        
    # Get email headers
    headers = msg.get('Headers')
    
    # Read email’s body
    body = str(msg.get_payload())
    
    # Check if body contains HTML tags
    if '<' in body and '>' in body:
        # Remove HTML tags if it's valid markup
        body = BeautifulSoup(body, features='html.parser').get_text()
    return headers, body


ham_path = './dataset/easy_ham'
spam_path = './dataset/spam'

ham = []
spam = []

def email_to_dataframe(directory_path, lst):
    for file in os.listdir(directory_path):
        file_path = f"{directory_path}/{file}"
        X = extract_email(file_path)
        lst.append(X)

email_to_dataframe(ham_path, ham)
email_to_dataframe(spam_path, spam)

ham_0 = pd.Series(np.zeros(len(ham)))
spam_1 = pd.Series(np.ones(len(spam)))

df_ham = pd.DataFrame()
df_ham['email'] = ham
df_ham['spam (1) or ham (0)'] = ham_0

df_spam = pd.DataFrame()
df_spam['email'] = spam
df_spam['spam (1) or ham (0)'] = spam_1

df = pd.concat([df_ham, df_spam], ignore_index=True)
df['email'] = df['email'].astype(str)
df['spam (1) or ham (0)'] = df['spam (1) or ham (0)'].astype(int)
df['email'] = df['email'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', regex=True)


print(df.head())



tf_idf = TfidfVectorizer(stop_words='english')

bow = tf_idf.fit_transform(df['email']).toarray()

vect = pd.DataFrame(bow, columns=tf_idf.get_feature_names())
vect["Spam (1) or ham (0)"] = df["Spam (1) or ham (0)"]


X_train, X_test, y_train, y_test = train_test_split(vect.iloc[:,:-1], vect['Spam (1) or ham (0)'], test_size = 0.25)


clf = SGDClassifier()

clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

conf_mat_train = confusion_matrix(y_train, y_pred_train)
conf_mat_test = confusion_matrix(y_test, y_pred_test)

results = pd.DataFrame(index = None)
results['Metric'] = ['Precision', 'Recall', 'Accuracy']
results['Classifier Train'] = [(precision_score(y_train, y_pred_train)), (recall_score(y_train, y_pred_train)), (clf.score(X_train, y_train))]
results['Classifier Test'] = [(precision_score(y_test, y_pred_test)), (recall_score(y_test, y_pred_test)), (clf.score(X_test, y_test))]

print(results.to_string(index=False))


def confusion_matrix_plot(conf_mat):
    n = len(conf_mat)
    plt.imshow(conf_mat, cmap='Blues', extent=[-0.5, n-0.5, -0.5, n-0.5])
    
    for i in range(n):
        for j in range(n):
            plt.text(i, j, conf_mat[n-j-1, i], ha='center', va='center')
    
    plt.colorbar()
    plt.xticks(range(n))
    plt.yticks(range(n), range(n-1, -1, -1))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.plot



def plot_learning_curve(test_error, training_size):
    plt.plot(training_size, test_error)
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Test Accuracy")
    plt.plot


print("Train Confusion Matrix:")
confusion_matrix_plot(conf_mat_train)


print("Test Confusion Matrix:")
confusion_matrix_plot(conf_mat_test)