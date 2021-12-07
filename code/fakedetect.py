# -*- coding: utf-8 -*-
#Trabalho IA2 G2
#Alunos: Igor Brinker

import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from google.colab import drive

import re  # Processamentos de dados

import nltk
from nltk.corpus import stopwords # Import de dicionario de palavras mais usadas de uma determinada lingua
from nltk.tokenize import word_tokenize # Import para a tokenizacao de palavras
nltk.download('punkt')
nltk.download('stopwords')

!pip install --upgrade gensim
from gensim.parsing.preprocessing import remove_stopwords # Removedor de stopwords
import gensim.downloader

pre_ft_vectors = gensim.downloader.load("glove-wiki-gigaword-100") # Load de dicionario com as stopwords

# Importacao dos vetorizadores baseados em scikit-learn (CountVect., and TF-IDF Vect.)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer as cvect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as tfvect
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold as kfold

# Import da library spacy
import spacy
nlp = spacy.load('en_core_web_sm')

from wordcloud import WordCloud  # Import para realizar a visualizacao grafica dos dados

import warnings
warnings.filterwarnings('ignore')  # Ignorar os warnings, para deixar as outputs mais limpas

true_ds = 'True.csv'
fake_ds = 'Fake.csv'

# Import do drive (configuracao para usar no google colab no lugar do jupyter notebook)
import_from_drive = True

if import_from_drive:
  drive.mount('/content/drive')
  at = r'/content/drive/MyDrive/FakeDetect/'
else:
  at = '/'

# True news
true_df = pd.read_csv(at + true_ds)

# Fake news
fake_df = pd.read_csv(at + fake_ds)

# Configuracao para apenas processar parte dos dados, para aumentar a velocidade de verificacao
p = 0.2  # Pacotes de dados com apenas 20% dos dados
true_df = true_df.head(int(p * true_df.shape[0]))
fake_df = fake_df.head(int(p * fake_df.shape[0]))

def clean_string(sent):
    sent = sent.lower()
    sent = re.sub('\n|\r|\t', '', sent)   # Tirar espacos em branco
    sent = re.sub(r'[^\w\s]+', '', sent)  # Remover pontuacoes
    return sent

def preprocess(df):
    df.dropna(subset = ['title', 'text'], inplace = True)  # Remover linhas da tabela com valores em branco
    vfunc = np.vectorize(clean_string)    # Aumentar velocidade de limpeza com vetorizacao
    df['title'] = vfunc(df['title'])
    df['text'] = vfunc(df['text'])
    return df

# Pre-processamento dos dados
true_df = preprocess(true_df)
fake_df = preprocess(fake_df)

# Usando o wordclous para criar imagens com os titulos
def disp_wordcloud(df):
    stop_words = nltk.corpus.stopwords.words('english')
    wc = WordCloud(max_words=250, stopwords=stop_words).generate(' '.join(df['title'].tolist()))
    plt.figure(figsize=(10,10))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    print()

print('\nTitulos onde mais se encontram noticias verdadeiras:\n')
disp_wordcloud(true_df)

print('\nTitulos onde mais se encontram fakenews:\n')
disp_wordcloud(fake_df)

# Cria um novo campo armazenando o numero de caracteres de cada campo dos artigos
def add_char_count_col(df, field):
    df['char_count_' + field] = [len(entry) for entry in df[field].to_list()]

def disp_histogram_char_count(df, field, news_type):
    ax = df['char_count_' + field].plot(kind='hist', legend=False,
                    title=f'Distribuicao de caracteres em {news_type} baseado no campo: {field}')
    ax.set_xlabel('Caracteres')
    ax.set_ylabel('Noticias')
    plt.xticks(rotation=0)
    plt.show()
    print()

def plot_average_char_count(true_df, fake_df, field):
    f = fake_df['char_count_' + field].mean()  # Media de characteres por campo
    t = true_df['char_count_' + field].mean()
    df = pd.DataFrame([['True', int(t)], ['Fake', int(f)]], columns=['News', 'Characters'])
    df.plot(x='News', y='Characters', kind='bar', width=0.3, xlabel='Type of News', legend=False,
                    ylabel='Chars', title=f'Media de caracteres em cada tipo de noticia baseado no campo: {field}')         
    plt.xticks(rotation=0)
    plt.show()
    print()

def char_stats(true_df, fake_df, field):
    add_char_count_col(true_df, field)  # Contador de caracteres por cada dataset
    add_char_count_col(fake_df, field)
    disp_histogram_char_count(true_df, field, 'True-news')
    disp_histogram_char_count(fake_df, field, 'Fake-news')
    plot_average_char_count(true_df, fake_df, field)
    true_df.drop('char_count_' + field, axis=1, inplace=True)  # Remove colunas auxiliares
    fake_df.drop('char_count_' + field, axis=1, inplace=True)

print('\nMostrando o status baseado no campo de Titulos\n')
char_stats(true_df, fake_df, 'title')

print('\nMostrando graficos com o campo de texto incluido\n')
char_stats(true_df, fake_df, 'text')

# Criacao de um novo campo para guardar o total de palavras de cada artigo
def add_word_count_col(df, field):
    df['word_count_' + field] = [len(entry.split()) for entry in df[field].to_list()]

def disp_histogram_word_count(df, field, news_type):
    ax = df['word_count_' + field].plot(kind='hist', legend=False,
                    title=f'Numero de palavras em {news_type} baseado no campo: {field}')
    ax.set_xlabel('Palavras')
    ax.set_ylabel('Noticias')
    plt.xticks(rotation=0)
    plt.show()
    print()


def plot_average_word_count(true_df, fake_df, field):
    f = fake_df['word_count_' + field].mean()
    t = true_df['word_count_' + field].mean()
    df = pd.DataFrame([['True', int(t)], ['Fake', int(f)]], columns=['News', 'Words'])
    df.plot(x='News', y='Words', kind='bar', width=0.3, xlabel='Type of News', legend=False,
                    ylabel='Num. of Words', title=f'Media de palavras em cada noticia, baseado no campo: {field}')         
    plt.xticks(rotation=0)
    plt.show()
    print()

def word_stats(true_df, fake_df, field):
    add_word_count_col(true_df, field)
    add_word_count_col(fake_df, field)
    disp_histogram_word_count(true_df, field, 'True-news')
    disp_histogram_word_count(fake_df, field, 'Fake-news')
    plot_average_word_count(true_df, fake_df, field)
    true_df.drop('word_count_' + field, axis=1, inplace=True)
    fake_df.drop('word_count_' + field, axis=1, inplace=True)

print('\nMostrando graficos com o campo de Titulo incluido\n')
word_stats(true_df, fake_df, 'title')

print('\nMostrando graficos com o campo de texto incluido\n')
word_stats(true_df, fake_df, 'text')

def remove_stopwords_from_str(string):
    return "".join(remove_stopwords(string))

# Criacao de um novo campo para guardar o total de palavras de cada artigo, apenas tirando palavras vazias (palavras mais comuns da lingua), para existir um treinamento mais preciso
def add_word_count_col_no_stopwords(df, field):
    df['word_count_' + field] = [len(remove_stopwords_from_str(entry).split()) for entry in df[field].to_list()]

def word_stats_no_stopwords(true_df, fake_df, field):
    add_word_count_col_no_stopwords(true_df, field)
    add_word_count_col_no_stopwords(fake_df, field)
    disp_histogram_word_count(true_df, field, 'True-news')
    disp_histogram_word_count(fake_df, field, 'Fake-news')
    plot_average_word_count(true_df, fake_df, field)
    true_df.drop('word_count_' + field, axis=1, inplace=True)  # Remocao de colunas auxiliares
    fake_df.drop('word_count_' + field, axis=1, inplace=True)

print('\nMostrando graficos com o campo titulo incluido, removendo palavras vazias\n')
word_stats_no_stopwords(true_df, fake_df, 'title')

print('\nMostrando graficos com o campo de texto incluido, removendo palavras vazias\n')
word_stats_no_stopwords(true_df, fake_df, 'text')

def lemmatize(string):
  lem = [token.lemma_ for token in nlp(string, disable=["parser", "ner"]) if token.lemma_ != '-PRON-']
  string = " ".join(lem)
  return string

# Lematizacao de todos os titulos e descricoes, para melhorar o treinamento da ia
def lemmatize_title_text(df):
    df_lem = df.copy()
    vfunc = np.vectorize(lemmatize)    # Aumento da velocidade de lematizacao via vetorizacao
    df_lem['text']  = vfunc(df_lem['text'])
    df_lem['title'] = vfunc(df_lem['title'])
    return df_lem

# Extraindo descricoes e titulos, para junta-los e colocar em um campo de strings individual
def extract_title_text(df):
    texts = " ".join(df['text'].to_list()).strip()
    titles = " ".join(df['title'].to_list()).strip()
    return (titles, texts)

def lemmatize_and_extract(df):
    lem_df = lemmatize_title_text(df)
    titles_texts = extract_title_text(lem_df)
    return lem_df, titles_texts

true_df_lem, true_tt = lemmatize_and_extract(true_df)
fake_df_lem, fake_tt = lemmatize_and_extract(fake_df)

# Criacao de um dataframe
def create_bow_df(tt, stopw):
    bigr = cvect(ngram_range=(2, 2), stop_words=stopw, min_df=2, max_df=9999)
    transf = bigr.fit_transform(tt)
    bigr_df = pd.DataFrame(transf.toarray(), columns = bigr.get_feature_names())
    return bigr_df

# Criacao de um dataframe de frequencia
def get_bigrams_freq(bigr_df, field) -> dict:
    index = 0 if field == 'title' else 1
    freq = bigr_df.iloc[index].to_list()
    bigr = bigr_df.columns.to_list()
    freq = list(zip(bigr, freq))
    return dict(freq)

def disp_wordcloud_bigr(freq, max_words_wc = 20):
    wc = WordCloud(max_words = max_words_wc).generate_from_frequencies(freq)
    plt.figure(figsize=(10,10))
    plt.imshow(wc)      # Inicializacao do display grafico
    plt.axis('off')
    plt.show()
    print()

true_bigr = create_bow_df(true_tt, stopwords.words('english'))
fake_bigr = create_bow_df(fake_tt, stopwords.words('english'))

true_title_bigr_freq = get_bigrams_freq(true_bigr, 'title')
true_text_bigr_freq = get_bigrams_freq(true_bigr, 'text')
fake_title_bigr_freq = get_bigrams_freq(fake_bigr, 'title')
fake_text_bigr_freq = get_bigrams_freq(fake_bigr, 'text')

topk = 20
print(f'\nTop-{topk} palavras encontradas em noticias verdadeiras')
disp_wordcloud_bigr(true_title_bigr_freq, topk)

topk = 20
print(f'\nTop-{topk} palavras encontradas em noticias falsas')
disp_wordcloud_bigr(true_text_bigr_freq, topk)

topk = 20
print(f'\nTop-{topk} palavras encontradas em titulos de noticias verdadeiras')
disp_wordcloud_bigr(fake_title_bigr_freq, topk)

topk = 20
print(f'\nTop-{topk} palavras encontradas em titulos de noticias falsas')
disp_wordcloud_bigr(fake_text_bigr_freq, topk)

# Merge nos campos "title" e "text" no field "content"
def merge_title_text(df):
    content = zip( df['title'].to_list(), df['text'].to_list() )
    df['content'] = [ str(title + text) for title, text in content ]    

# Cria o campo "content" e "label", para dar o drop em outros campos pre-existentes
def prepare_df(df, label):
    df['label'] = [ label for i in range(df.shape[0]) ]  # Add label
    merge_title_text(df)
    df.drop(columns = ['title', 'text', 'subject', 'date'], inplace=True)

prepare_df(true_df_lem, 1)
prepare_df(fake_df_lem, 0)

true_df_lem.head()

# Divide o dataframe em teste e treinamento
combined = pd.concat([true_df_lem, fake_df_lem])
train_df, test_df = train_test_split(combined, train_size=0.5, random_state=420)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Guarda os dados de teste e treinamento em dois arquivos .csv distintos
test_df.to_csv (path_or_buf='test.csv',  columns=test_df.columns)
train_df.to_csv(path_or_buf='train.csv', columns=train_df.columns)

train_df.tail()

test_df.tail()

# Representacao do vetorizador
def get_representation(vectorizer, train_df, test_df):
    # pega os labels de teste e treinamento
    y_train = train_df['label']
    y_test = test_df['label']
    # Vetorizacao do treinamento e teste.
    # O uso do fit_transform() no treinamento, serve para recolher parametros ao mesmo tempo que mais dados sao processados
    x_train = vectorizer.fit_transform(train_df['content'])
    x_test = vectorizer.transform(test_df['content'])
    return x_train, y_train, x_test, y_test

cv_rep = get_representation(CountVectorizer(), train_df, test_df)
tf_rep = get_representation(TfidfVectorizer(), train_df, test_df)

# Transfere os dados para ser realizado o calculo de precisao
def __classify(method, x_train, y_train, x_test, y_test, wmean = False):
    # Faz o escalonamento com a media dos dados
    pipe = make_pipeline(StandardScaler(with_mean=wmean), method)
    # Aplica os dados de treino no escalonamento
    pipe.fit(x_train, y_train)  
    # Aplica os dados de teste, sem os dados de treino
    pipe.score(x_test, y_test)  

    y_pred = pipe.predict(x_test)
    # Print do resumo de dados
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred);
    print('Precisao de {:.2f} %'.format(accuracy*100))

# Transfere os dados para a library realizar o calculo de precisao
def classify(method, cv_rep, tf_rep, tfid = False, toarr = False):
    # Extracao de sets
    x_train, y_train, x_test, y_test = tf_rep if tfid else cv_rep
    # Converte o array se necessario
    if toarr: x_train, x_test = x_train.toarray(), x_test.toarray()

    __classify(method, x_train, y_train, x_test, y_test, False)

def sim_logistic_regression(cv_rep, tf_rep, tfid = False):
    classify(LogisticRegression(), cv_rep, tf_rep, tfid, False)

sim_logistic_regression(cv_rep, tf_rep, False)

# Aplica o algoritmo de regressao logistica para treinamento (predicao de valores tomados)
sim_logistic_regression(cv_rep, tf_rep, True)

# Aplica a biblioteca para realizar treino via calculo de Naive Bayes (classificacao probabilistica)
def sim_naive_bayes(cv_rep, tf_rep, tfid = False):
    classify(GaussianNB(), cv_rep, tf_rep, tfid, True)

sim_naive_bayes(cv_rep, tf_rep, False)

sim_naive_bayes(cv_rep, tf_rep, True)

# Treinando via calculo de dispercao
def sim_svc(parameters, cv_rep, tf_rep, tfid = False):
    classify(GridSearchCV(SVC(), parameters), cv_rep, tf_rep, tfid, False)

parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}

sim_svc(parameters, cv_rep, tf_rep, False)

sim_svc(parameters, cv_rep, tf_rep, True)

# Criando uma random forest para tratar os resultados de treinamento, onde multiplas arvores de decisoes sao criadas no tempo de treinamento
def sim_random_forest(cv_rep, tf_rep, tfid = False):
    classify(RandomForestClassifier(), cv_rep, tf_rep, tfid, False)

sim_random_forest(cv_rep, tf_rep, False)

sim_random_forest(cv_rep, tf_rep, True)