#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[64]:


# # Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[65]:


countries = pd.read_csv("countries.csv", decimal=',')


# In[66]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# In[136]:


countries['Climate'].value_counts()


# In[137]:


countries['Region'].value_counts()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.info()


# In[6]:


countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[7]:


def q1():
    """Answer of question 01
    
    Returns
    -------
    list
        Regions name sorted by alphabetical order
    """
    
    return list(np.sort(countries['Region'].unique()))


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[129]:


def q2():
    """Answer of question 02
    
    Returns
    -------
    int
        Number of countries above the 90º percentile
    """
    
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    discretizer.fit(countries[['Pop_density']])
    score_bins = discretizer.transform(countries[['Pop_density']])
    
    return int(sum(score_bins[:, 0] >= 9))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[158]:


def q3():
    """Answer of question 03
    
    Returns
    -------
    int
        Number of new features created by one-hot encoding
    """
    
    return pd.get_dummies(countries[['Climate', 'Region']].fillna('None'), 
                          columns=['Climate', 'Region']).shape[1]


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[11]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[50]:


def q4():
    """Answer of question 04
    
    Returns
    -------
    float
        Value of Arable variable after 
    """
    
    # Get numerical columns
    num_var = list(countries.select_dtypes(['int64', 'float64']).columns)
    # Prepare pipeline to fill missing values with median and than standardize them
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('standardization', StandardScaler())
    ])
    pipeline_transform = pipeline.fit_transform(countries[num_var])
    pipeline_df = pd.DataFrame(pipeline_transform, columns=num_var)
    
    test_country_df = pd.DataFrame([test_country], columns=countries.columns)
    test_country_transformed = pipeline.transform(test_country_df[num_var])
    arable_transformed = pd.DataFrame(test_country_transformed, columns=num_var)['Arable']
    
    return float(arable_transformed.round(3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[68]:


sns.boxplot(countries['Net_migration'], orient='vertical')
plt.show()


# In[69]:


def q5():
    """Answer of question 05
    
    Returns
    -------
    tuple
        Number of inferior and superior outliers and if they should be removed
        in the format (n_inferior: int, n_superior: int, remove: bool)
    """
    
    # First quartile
    Q1 = countries['Net_migration'].quantile(.25)
    # Third quartile
    Q3 = countries['Net_migration'].quantile(.75)
    # Interquartile range
    IQR = Q3 - Q1
    
    n_inferior = countries[countries['Net_migration'] < Q1 - 1.5 * IQR].shape[0]
    n_superior = countries[countries['Net_migration'] > Q3 + 1.5 * IQR].shape[0]
    
    return tuple([n_inferior, n_superior, False])


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[126]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories,
                                   shuffle=True, random_state=42)


# In[123]:


def q6():
    """Answer of question 06
    
    Returns
    -------
    int
        Number of times that 'phone' appeared
    """
    
    vectorizer = CountVectorizer()
    newsgroups_count = vectorizer.fit_transform(newsgroups['data'])
    
    word_list = vectorizer.get_feature_names()
    count_list = np.array(newsgroups_count.sum(axis=0)).reshape(-1)
    word_count = dict(zip(word_list, count_list))
    
    return int(word_count['phone'])


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[127]:


def q7():
    """Answer of question 07
    
    Returns
    -------
    float
        TF-IDF of 'phone' word
    """
    
    vectorizer = TfidfVectorizer()
    newsgroups_count = vectorizer.fit_transform(newsgroups['data'])
    
    word_list = vectorizer.get_feature_names()
    count_list = np.array(newsgroups_count.sum(axis=0)).reshape(-1)
    word_tdidf = dict(zip(word_list, count_list))
    
    return float(word_tdidf['phone'].round(3))

