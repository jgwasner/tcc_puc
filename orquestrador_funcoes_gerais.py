
# Bibliotecas

import pandas as pd
import numpy as np
import csv
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from string import punctuation
import re
from unicodedata import normalize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import json
import codecs
import os
import glob
from nltk.stem.snowball import PortugueseStemmer

#import spacy 



# Inicialização

#nlp = spacy.load("pt_core_news_sm")

nome_arquivo_configuracao = 'config.json'
tokenizer = RegexpTokenizer(r'\w+')

stemmer = nltk.stem.RSLPStemmer()
#stemmer = PortugueseStemmer()

lista_stopwords = set(stopwords.words('portuguese') + list(punctuation))


sinonimos = {'18':'dezoito', 'tg': 'tiro de guerra', 'tatoo':'tatuagem', 'smo':'servico militar obrigatorio','2a':'segunda',
            'pcd':'pessoa com deficiencia', 'cdi':'certificado de dispensa de incorporacao',
            'npor': 'nucleo de preparacao de oficiais da reserva', 'cpor':'centro de preparacao de oficiais da reserva',
            'crdi':'certidao de registro de dados individuais','ctsm':'certidao de tempo de servico militar',
            '2ª':'segunda', 'coronavirus':'covid', 'corona virus':'covid', 'chrome':'navegador', 'mozilla':'navegador',
             'firefox':'navegador', 'safari':'navegador', 'opera':'navegador','internet exporer':'navegador', 
             'linux':'sistema operacional', 'android':'sistema operacional', 'ios':'sistema operacional', 
             'windows':'sistema operacional','bb':'banco do brasil', 'cef': 'caixa economica federal',
             'mei': 'microempreendedor individual', 'pgd':'programa gerador de declaracao', '13': 'decimo terceiro', 
             'e-cac':'cac','pg':'programa gerador'}

# Funções

def print_dict(dict ,ident_str='  ', ident=0):
    try:
        dict.keys()
    except:
        print(ident_str * ident + key + ':',dict[key])
        return
    
    for key in dict.keys():
        try:
            subkeys = dict[key].keys()
            print(ident_str * ident + key + ':')
            print_dict(dict[key], ident_str=ident_str, ident=ident+1)
        except:
            print(ident_str * ident + key + ':',dict[key])
        finally:
            if ident == 0:
                print('-'*120)
    return

def carregar_configuracoes():
    cfg = json.loads(open(nome_arquivo_configuracao).read())
    print('Conferir as configurações antes de prosseguir')
    print_dict(cfg, ident_str='  ')
    return cfg


def remover_tags_html(texto):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', texto)
 
def limpar_texto(texto, aplicar_stemmer):

    # Converte para minúsculas
    texto = texto.lower()
    
    # Remove acentos
    texto = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    
    # Substitui algumas palavras e siglas por sinônimos, e certos números especiais por literais.
    for sinonimo in sinonimos:
        texto = re.sub(r'\b%s\b' % sinonimo, sinonimos[sinonimo], texto)
     
    # Remove um caso comum de apóstrofo 
    texto = texto.replace("d'", " ")
    
    # Remove Tags HTML
    texto = remover_tags_html(texto)  
    
    # Remove URLs
    texto = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', texto, flags=re.MULTILINE)
    
    # Remove emails   
    texto = re.sub(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+', ' ', texto, flags=re.MULTILINE)  
    
    # Converte R$ em "reais"
    texto = texto.replace('r$','reais')
    
    # Remove caracteres especiais e números
    texto = re.sub(r'[^a-zA-Z]',' ',texto)
     
    # Remove espaços extras e palavras menores do que um tamanho mínimo
    tokens = tokenizer.tokenize(texto)
    tokens = [palavra for palavra in tokens if len(palavra) > 1]          
    tokens = [palavra for palavra in tokens if palavra not in lista_stopwords] 
    
    if aplicar_stemmer:
        for i in range(len(tokens)):
            token = stemmer.stem(tokens[i])
            tokens[i] = token 
            
    texto = ' '.join(tokens)  
    
    return texto


# Calcula métricas de desempenho do classificador.
def get_metrics(y_test, y_predicted, decimais=4, verbose=True): 
    
    accuracy = round(accuracy_score(y_test, y_predicted),decimais)
    
    precision = round(precision_score(y_test, y_predicted, pos_label=1, average='macro', zero_division=0),decimais)             
    recall = round(recall_score(y_test, y_predicted, pos_label=1, average='macro', zero_division=0),decimais) 
    f1_macro = round(f1_score(y_test, y_predicted, pos_label=1, average='macro', zero_division=0),decimais)
    f1_weighted = round(f1_score(y_test, y_predicted, pos_label=1, average='weighted', zero_division=0),decimais)
 
    if verbose:
        print('Acurácia:',accuracy,'- Precisão (Macro):',precision,'- Recall (Macro):',recall,
              '- F1 (Macro):',f1_macro, '- F1 (Weighted):',f1_weighted,)
    
    return accuracy, precision, recall, f1_macro


def executa_grid_search(param_grid, clf, X_GS,  y_GS, vectorizer, scoring='f1_macro'):
    classif_pipe = Pipeline([('vectorize', vectorizer),
                           ('classifier', clf)])  
    grid_search = GridSearchCV(classif_pipe, param_grid, cv=3, n_jobs=-1, verbose=3, 
                               scoring=scoring)
    grid_search.fit(X_GS, y_GS)
    results = pd.DataFrame(grid_search.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    results.reset_index(inplace=True)
    
    print(grid_search.best_estimator_['classifier'].__class__.__name__,
          '- Média Score:',results.iloc[0]['mean_test_score'],
          '\nParams:',results.iloc[0]['params'])
    
    return grid_search.best_estimator_, results


def print_progress_bar(iteration, total, prefix = '', suffix = '',
                       decimals = 1, length = 50, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()