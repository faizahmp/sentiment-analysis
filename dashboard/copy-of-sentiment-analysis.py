# Libraries
import streamlit as st
from joblib import load
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
import torch
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import warnings
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
import numpy as np
import random
from random import sample
import time
import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
# WordCloud
warnings.filterwarnings("ignore")

st.title('Hasil Analisis Sentimen Aksi Demokrasi di Indonesia pada Komentar Media Sosial YouTube')
################# Model Testing Sentiment Result #################
df_merged = pd.read_csv(
    "https://raw.githubusercontent.com/faizahmp/sentiment-analysis/main/data/df_merged.csv")

st.header('Hasil Sentimen dengan Model')
labels = ['Negatif', 'Positif']
sizes = df_merged['sentiment'].value_counts()
colors = ['#ff4747', '#07da63']
fig, ax = plt.subplots()
ax.pie(sizes, colors=colors, autopct='%1.1f%%')
plt.legend(labels, loc=1)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal')
plt.tight_layout()
st.pyplot(fig)

################# Word Cloud #################
st.header('Word Cloud')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Komentar Positif')
    positive_text = ' '.join(df_merged[df_merged['sentiment'] == 1]['text'])

    wordcloud = WordCloud(max_font_size=50, max_words=100,
                          background_color="white").generate(positive_text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
with col2:
    st.subheader('Komentar Negatif')
    negative_text = ' '.join(df_merged[df_merged['sentiment'] == 0]['text'])

    wordcloud = WordCloud(max_font_size=50, max_words=100,
                          background_color="white").generate(negative_text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

################ Frekuensi Kata ##################
st.subheader('Frekuensi Kata')
# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
# words = ['demo', 'mahasiswa', 'jokowi', 'bbm', 'indonesia',
#          'merdeka', 'maju', 'pemerintah', 'turun', 'harga']
# values = [1530, 1363, 1035, 943, 876, 800, 764, 499, 456, 401]
# ax.bar(words, counts, color="#4361EE")
# ax.set_xticks(words)
# ax.set_xlabel("Words")
# ax.set_xticklabels(words, rotation=45, ha="right")
# ax.set_ylabel("Frequency")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0, 0, 1, 1])
all_words = ' '.join(df_merged['text'])
tokens = word_tokenize(all_words)
word_counts = Counter(tokens)  # ini
most_common_words = word_counts.most_common(10)
words = [word[0] for word in most_common_words]
counts = [word[1] for word in most_common_words]

ax.bar(words, counts, color="#4361EE")
ax.set_xticks(words)
ax.set_xlabel("Words")
plt.title('Top 10 Most Frequent Words')
ax.set_xticklabels(words, rotation=45, ha="right")
ax.set_ylabel("Frequency")

# plt.figure(figsize=(10, 6))
# plt.bar(words, counts, color='skyblue')
# plt.xlabel('Words')
# plt.ylabel('Frequency')
# plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
st.pyplot(fig)


############### MODELLING ######################

##### SVM #######
# feature extraction  / Tokenization (Word2Vec)
df_merged_1 = pd.read_csv(
    'https://raw.githubusercontent.com/faizahmp/sentiment-analysis/main/data/df_merged_cut.csv')
load_model = load('model/svm_model.joblib')
X_train, X_test, y_train, y_test = train_test_split(
    df_merged_1['text'], df_merged_1['sentiment'], test_size=0.2, random_state=42)
y_pred_svm = load_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)

##### NAIVE BAYES #######
load_model = load('model/mnb_model.joblib')
X_train, X_test, y_train, y_test = train_test_split(
    df_merged_1['text'], df_merged_1['sentiment'], test_size=0.2, random_state=42)
y_pred_nb = load_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)

##### INDOBERT #######
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('there are %d GPU(s) available.' % torch.cuda.device_count())
#     print('we will use the GPU: ', torch.cuda.get_device_name(0))
# else:
#     print("No GPU available, using the CPU instead")
#     device = torch.device("cpu")

# model = BertForSequenceClassification.from_pretrained("indolem/indobert-base-uncased",
#                                                       num_labels=2,
#                                                       output_attentions=False,
#                                                       output_hidden_states=False)
# tokenizer = BertTokenizer.from_pretrained(
#     'indolem/indobert-base-uncased', do_lower_case=True)

# model.load_state_dict(torch.load(
#     'indobert_model_sentiment.pth', map_location=torch.device('cpu')))
# model.eval()

# df_merged = df_merged.drop(columns=['original_text'])
# df_merged = df_merged.rename(columns={'sentiment': 'label'})
# df_merged.head()

# text = df_merged['text'].values
# labels = df_merged['label'].values

# input_ids = []

# for sentence in text:
#     encoded_sentences = tokenizer.encode(sentence, add_special_tokens=True)
#     input_ids.append(encoded_sentences)

# max_len = 64
# input_ids = pad_sequences(input_ids, maxlen=max_len,
#                           dtype='long', value=0, truncating='post', padding='post')

# attention_mask = []

# for sentence in input_ids:
#     att_mask = [int(token_id > 0) for token_id in sentence]
#     attention_mask.append(att_mask)

# train_input, test_input, train_labels, test_labels = train_test_split(input_ids,
#                                                                       labels,
#                                                                       random_state=2017,
#                                                                       test_size=0.2)
# train_mask, test_mask, _, _ = train_test_split(attention_mask,
#                                                labels,
#                                                random_state=2017,
#                                                test_size=0.2)

# train_input, validation_input, train_labels, validation_labels = train_test_split(train_input,
#                                                                                   train_labels,
#                                                                                   random_state=2018,
#                                                                                   test_size=0.15)
# train_mask, validation_mask, _, _ = train_test_split(train_mask,
#                                                      train_mask,
#                                                      random_state=2018,
#                                                      test_size=0.15)

# test_input = torch.tensor(test_input)
# test_labels = torch.tensor(test_labels)
# test_mask = torch.tensor(test_mask)

# test_data = TensorDataset(test_input, test_mask, test_labels)
# test_sampler = SequentialSampler(test_data)
# test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

# predictions = []
# true_labels = []

# for batch in test_dataloader:
#     # batch = tuple(t.to(device) for t in batch)
#     b_input_ids, b_input_mask, b_labels = batch

#     with torch.no_grad():
#         outputs = model(b_input_ids,
#                         token_type_ids=None,
#                         attention_mask=b_input_mask)

#     logits = outputs[0]

#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()

#     predictions.append(logits)
#     true_labels.append(label_ids)

# flat_predictions = [item for sublist in predictions for item in sublist]
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# flat_true_labels = [item for sublist in true_labels for item in sublist]

# idbert_acc = accuracy_score(flat_true_labels, flat_prediction)
# idbert_precision = precision_score(flat_true_labels, flat_prediction)
# idbert_recall = recall_score(flat_true_labels, flat_prediction)

idbert_acc = 0.6882984159427695
idbert_precision = 0.6743295019157088
idbert_recall = 0.4444444444444444

################# Performance Comparison #################
# set the data

models_performance = {
    'IndoBERT': {'accuracy': idbert_acc, 'precision': idbert_precision, 'recall': idbert_recall},
    'Naive Bayes': {'accuracy': nb_accuracy, 'precision': nb_precision, 'recall': nb_recall},
    'SVM': {'accuracy': svm_accuracy, 'precision': svm_precision, 'recall': svm_recall}
}

best_accuracy_values = None
best_precision_values = None
best_recall_values = None

for model_name, metrics in models_performance.items():
    if best_accuracy_values is None or metrics['accuracy'] > best_accuracy_values:
        best_accuracy_values = metrics['accuracy']
    if best_precision_values is None or metrics['precision'] > best_precision_values:
        best_precision_values = metrics['precision']
    if best_recall_values is None or metrics['recall'] > best_recall_values:
        best_recall_values = metrics['recall']

st.header('Komparasi Akurasi')
col1, col2, col3 = st.columns(3)

with col1:
    accuracy = str(round(best_accuracy_values*100, 2))+"%"
    st.metric("Accuracy", value=accuracy)

with col2:
    precision = str(round(best_precision_values*100, 2)) + "%"
    st.metric("Precision", value=precision)

with col3:
    recall = str(round(best_recall_values*100, 2)) + "%"
    st.metric("Recall", value=recall)

x = np.arange(3)
indobert = [round(idbert_acc * 100, 2), round(idbert_precision * 100, 2),
            round(idbert_recall * 100, 2)]
nb = [round(nb_accuracy * 100, 2), round(nb_precision * 100, 2),
      round(nb_recall * 100, 2)]
svm = [round(svm_accuracy * 100, 2), round(svm_precision * 100, 2),
       round(svm_recall * 100, 2)]
width = 0.2

# set the pastel colors
pastel_green = "#39b54a"
pastel_blue = "#87cefa"
pastel_yellow = "#fff44f"

# create the figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# plot the data in grouped manner of bar type
ax.bar(x - 0.2, indobert, width, color=pastel_green, label="IndoBERT")
ax.bar(x, nb, width, color=pastel_blue, label="Naive Bayes")
ax.bar(x + 0.2, svm, width, color=pastel_yellow, label="SVM")

# set the x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall"])

# set the y-axis label
ax.set_ylabel("Scores")

# add a legend
ax.legend(["IndoBERT", "Naive Bayes", "SVM"], loc=1,
          frameon=True, edgecolor="black", borderpad=0.1)

st.pyplot(fig)

################## SIDEBAR ##################
lst = ['[Video 1](https://www.youtube.com/watch?v=0aJv3qiAA18)',
       '[Video 2](https://www.youtube.com/watch?v=LvRmkq70AzI)',
       '[Video 3](https://www.youtube.com/watch?v=d_ogyVC1Ybk)',
       '[Video 4](https://www.youtube.com/watch?v=mtu-Xa4dNEk)',
       '[Video 5](https://www.youtube.com/watch?v=ziRu2zkffq0)']

s = ''

for i in lst:
    s += "- " + i + "\n"


with st.sidebar:
    st.subheader('Tentang Dataset')
    st.markdown("""Dataset ini merupakan data komentar aksi demonstrasi di Indonesia
                di platform YouTube pada 5 video yang diunggah dari rentang Oktober
                2020 hingga September 2022. """)

    st.subheader('Sumber Dataset')

    st.markdown(s)
