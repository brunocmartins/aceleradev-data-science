#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

import pandas as pd
import numpy as np

# Import black friday data
black_friday = pd.read_csv("black_friday.csv")

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

def q1():
    """Answer of question 01
    
    Returns
    -------
    tuple
        A tuple of number of observations and number of columns as
        followed: (n_observations, n_columns)
    """
    
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

def q2():
    """Answer of question 02
    
    Returns
    -------
    int
        Number of women aged between 26 and 35 years
    """
    
    women = black_friday['Gender'] == 'F'
    age_26_35 = black_friday['Age'] == '26-35'
    
    return black_friday[women & age_26_35].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

def q3():
    """Answer of question 03
    
    Returns
    -------
    int
        Number of unique users
    """
    
    return black_friday['User_ID'].drop_duplicates().shape[0]


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

def q4():
    """Answer of question 04
    
    Returns
    -------
    int
        Number of different column types
    """
    
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

def q5():
    """Answer of question 05
    
    Returns
    -------
    float
        Percentage of lines with at least one NaN value
    """
    
    n_records = black_friday.shape[0]
    non_nan = black_friday.dropna().shape[0]
    
    return 1 - (non_nan / n_records)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

def q6():
    """Answer of question 06
    
    Returns
    -------
    int
        Number of missing data on the column with the most NaN values
    """
    
    n_nulls = []
    for column in black_friday.columns:
        n = black_friday[column].isna().sum()
        n_nulls.append(n)
    
    return int(max(n_nulls))


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

def q7():
    """Answer of question 07
    
    Returns
    -------
    int
        Most frequent value excluding missing data
    """
    
    return int(black_friday['Product_Category_3'].mode()[0])


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

def q8():
    """Answer of question 08
    
    Returns
    -------
    float
        Mean of 'Purchase' feature after normalization
    """
    
    purchase_min = black_friday['Purchase'].min()
    purchase_max = black_friday['Purchase'].max()
    
    purchase_norm = (black_friday['Purchase'] - purchase_min) / (purchase_max - purchase_min)
    
    return float(purchase_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

def q9():
    """Answer of question 09
    
    Returns
    -------
    int
        Number of values between -1 and 1 in 'Purchase' feature
        after standardization
    """
    
    purchase_mean = black_friday['Purchase'].mean()
    purchase_std = black_friday['Purchase'].std()
    
    purchase_stand = (black_friday['Purchase'] - purchase_mean) / purchase_std
    
    return purchase_stand.between(-1, 1).sum()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

def q10():
    """Answer of question 10
    
    Returns
    -------
    bool
        'False' if any missing observation in 'Product_Category_2'
        has an value in 'Product_Category_3' and 'True' if all missing
        data in 'Produc_Category_2' is also a missing data in 'Product_Category_3'
    """
    
    prod_2_null = black_friday['Product_Category_2'].isna()
    prod_3_not_null = black_friday['Product_Category_3'].notna()
    
    answer = black_friday[prod_2_null & prod_3_not_null].empty
    
    return answer