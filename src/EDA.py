# %%
import numpy as numpy
import matplotlib.pyplot as plt
import os
import pandas as pd

# %% [markdown]
# Metadados do dataset
# 

# %%
dataset = pd.read_csv("/home/wytcor/PROJECTs/mestrado-ufes/lab-life/datasets/zr7vgbcyr2-1/metadata.csv", sep=",")
dataset


# %% [markdown]
# Remoção de NaN

# %%
dataset=dataset.dropna()
dataset

# %% [markdown]
# Após a remoção dos dados com NaN, vamos remover as colunas de id's. Só depois vamos realizar a anáse dos dados

# %%
dataset_features=dataset.drop(columns=['patient_id', 'lesion_id', 'img_id'])
dataset_features

# %% [markdown]
# Análise das categorias

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['diagnostic'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas')
plt.xlabel('Tipo de mancha')
plt.ylabel('Quantidade')
plt.show()

# %% [markdown]
# Verificamos que há uma grande desequilíbrio entre entre a quantidade de amostras, o que poderá resultar em modelo final extremamente enviesado

# %% [markdown]
# Para as variáveis que foram a analisadas por biópsia

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['biopsed'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - biópsia')
plt.xlabel('Tipo de mancha')
plt.ylabel('Quantidade')
plt.show()

# %% [markdown]
# Fumantes

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['background_father'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(24, 12))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - background_father')
plt.xlabel('Backgroundliar familiar')
plt.ylabel('Quantidade')
plt.show()

# %% [markdown]
# O histórico da etnia do paciente

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['smoke'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - smoke')
plt.xlabel('Se fuma ou não')
plt.ylabel('Quantidade')
plt.show()

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['pesticide'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - pesticide')
plt.xlabel('Se está exposto(a) ao uso de pesticida ou não')
plt.ylabel('Quantidade')
plt.show()

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['gender'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - gender')
plt.xlabel('Sexo dos pacientes')
plt.ylabel('Quantidade')
plt.show()

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['cancer_history'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - cancer_history')
plt.xlabel('Se os pacientes possuem um histórico de ocorrência de câncer, sendo qualquer tipo de câncer considerado')
plt.ylabel('Quantidade')
plt.show()

# %%
# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['skin_cancer_history'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - skin_cancer_history')
plt.xlabel('Se os pacientes possuem um histórico de ocorrência de câncer de pele')
plt.ylabel('Quantidade')
plt.show()

# %%


# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['grew'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - grew')
plt.xlabel('Se foi observado que as manchas na pele aumentaram')
plt.ylabel('Quantidade')
plt.show()


# %%




# Para a análise dos tipos de diagnósticos
quality_counts = dataset_features['has_piped_water'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts)
plt.title('Quantidade dos tipos de variáveis a serem contadas - has_piped_water')
plt.xlabel('Se os pacientes possuem água encanada em casa')
plt.ylabel('Quantidade')
plt.show()


# %% [markdown]
# Podemos verificar que a maioria das amostras teve seu resultado final verificado por biópsia

# %% [markdown]
# Relação entre as variáveis - variáveis numéricas

# %%
import seaborn as sns

# Assuming 'df' is your DataFrame
plt.figure(figsize=(15, 10))

# Using Seaborn to create a heatmap
sns.heatmap(dataset_features.corr(), annot=True, fmt='.2f', cmap='YlGnBu', linewidths=2)

plt.title('Correlation Heatmap')
plt.show()



# %% [markdown]
# Nesta análise, apenas o diâmetro 2 e diâmetro 1 possui uma correlação significativa

# %% [markdown]
# Verificar a relação entre as variáveis categóricas

# %%
columns_original = list(dataset_features.columns)
columns_numericas = ['age', 'fitspatrick', 'diameter_1', 'diameter_2']
columns_categoricas = list(set(columns_original) - set(columns_numericas))
print("Colunas categóricas:", columns_categoricas)


# %% [markdown]
# Verificação da co-ocorência

# %%
# Loop sobre cada variável categórica em 'columns_categoricas'
for categorical_category in columns_categoricas:
    # Calcula a tabela cruzada normalizada entre a variável categórica e 'diagnostic'
    results = pd.crosstab(dataset_features[categorical_category], 
                          dataset_features['diagnostic'], 
                          normalize='index')
    
    # Exibe o nome da variável categórica e os resultados da tabela cruzada
    print(f"Tabela cruzada para '{categorical_category}' em relação a 'diagnostic':\n")
    print(results)
    print("\n" + "-"*50 + "\n")


# %% [markdown]
# Verificação do valor chi² e p, para saber a relação de uma com outra

# %%
from scipy.stats import chi2_contingency

for categorical_category in columns_categoricas:
    # Calcula o teste qui-quadrado
    print(f"Para a análise da variável categórica: {categorical_category}\n")
    tabela_contingencia = pd.crosstab(dataset_features[categorical_category], dataset_features['diagnostic'])
    chi2, p, _, _ = chi2_contingency(tabela_contingencia)
    print(f'Qui-quadrado: {chi2}, Valor-p: {p}')


# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Criar uma matriz para armazenar os valores de p
p_values = pd.DataFrame(index=columns_categoricas, columns=columns_categoricas)

# Loop para calcular o p-valor do teste qui-quadrado entre cada par de variáveis categóricas
for var1 in columns_categoricas:
    for var2 in columns_categoricas:
        if var1 != var2:
            # Criar uma tabela de contingência entre as duas variáveis
            contigency_table = pd.crosstab(dataset_features[var1], dataset_features[var2])
            # Calcular o teste qui-quadrado
            _, p, _, _ = chi2_contingency(contigency_table)
            # Armazenar o p-valor na matriz
            p_values.loc[var1, var2] = p
        else:
            # Colocar NaN na diagonal (mesma variável comparada consigo mesma)
            p_values.loc[var1, var2] = None

print("Matriz de p-valores do teste qui-quadrado entre variáveis categóricas:\n")
print(p_values)


# %% [markdown]
# # Conclusão
# 
# Temos a partir das análises acima que os fatores com maiores associações entre si são, não respctivamente:
#  - bleed
#  - background_mother
#  - background_father
#  - gender
#  - region
#  - grew
#  - smoke
#  - elevation 
#  - biopsed
#  - has_piped_water
#  - changed
#  - pesticide
#  - itch
# 
# 


