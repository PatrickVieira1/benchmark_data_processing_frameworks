import pandas as pd
from timeit_decorator import timeit_decorator
import os
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

start_time = time.time()

@timeit_decorator
def ler_dataframe_csv(file):
    df = pd.read_csv(file)
    return df

@timeit_decorator
def salvar_dataframe_csv(df, file):
    df.to_csv(file, index=False)

# Ler os arquivos CSV
dados_covid_cidades = ler_dataframe_csv('data/brazil_covid19_cities.csv')
dados_populacao = ler_dataframe_csv('data/brazil_population_2019.csv')

@timeit_decorator
def cidades_de_alto_impacto(dados_covid_cidades, dados_populacao):
    # Criar um mapeamento de nomes completos dos estados para abreviações
    mapeamento_estados = {
        'Acre': 'AC', 'Alagoas': 'AL', 'Amazonas': 'AM', 'Amapá': 'AP',
        'Bahia': 'BA', 'Ceará': 'CE', 'Distrito Federal': 'DF', 'Espírito Santo': 'ES',
        'Goiás': 'GO', 'Maranhão': 'MA', 'Minas Gerais': 'MG', 'Mato Grosso do Sul': 'MS',
        'Mato Grosso': 'MT', 'Pará': 'PA', 'Paraíba': 'PB', 'Pernambuco': 'PE',
        'Piauí': 'PI', 'Paraná': 'PR', 'Rio de Janeiro': 'RJ', 'Rio Grande do Norte': 'RN',
        'Rondônia': 'RO', 'Roraima': 'RR', 'Rio Grande do Sul': 'RS',
        'Santa Catarina': 'SC', 'Sergipe': 'SE', 'São Paulo': 'SP', 'Tocantins': 'TO'
    }
    
    # Renomear a coluna "estado" no DataFrame de população
    dados_populacao['state'] = dados_populacao['state'].replace(mapeamento_estados)
    
    dados_covid_cidades['city'] = dados_covid_cidades['name']
    
    # Converter a coluna "data" para o formato datetime
    dados_covid_cidades['date'] = pd.to_datetime(dados_covid_cidades['date'], format='%Y-%m-%d')
    
    # Realizar o join com os dados de população
    resultado = dados_covid_cidades.merge(dados_populacao, on=['city', 'state'], how='inner')
    
    # Adicionar uma coluna de taxa de mortalidade
    resultado['death_rate'] = resultado['deaths'] / resultado['population']
    
    # Calcular a média móvel de 7 dias para óbitos
    resultado['7_day_avg_deaths'] = (
        resultado.groupby('state')['deaths']
        .rolling(window=7).mean().reset_index(0, drop=True)
    )

    # Ordenar pelos valores de taxa de mortalidade em ordem decrescente
    resultado = resultado.sort_values(by='death_rate', ascending=False)
    return resultado

@timeit_decorator
def transformar_dados_covid(dados_covid_cidades, dados_populacao):
    # Criar um mapeamento de nomes completos dos estados para abreviações
    mapeamento_estados = {
        'Acre': 'AC', 'Alagoas': 'AL', 'Amazonas': 'AM', 'Amapá': 'AP',
        'Bahia': 'BA', 'Ceará': 'CE', 'Distrito Federal': 'DF', 'Espírito Santo': 'ES',
        'Goiás': 'GO', 'Maranhão': 'MA', 'Minas Gerais': 'MG', 'Mato Grosso do Sul': 'MS',
        'Mato Grosso': 'MT', 'Pará': 'PA', 'Paraíba': 'PB', 'Pernambuco': 'PE',
        'Piauí': 'PI', 'Paraná': 'PR', 'Rio de Janeiro': 'RJ', 'Rio Grande do Norte': 'RN',
        'Rondônia': 'RO', 'Roraima': 'RR', 'Rio Grande do Sul': 'RS',
        'Santa Catarina': 'SC', 'Sergipe': 'SE', 'São Paulo': 'SP', 'Tocantins': 'TO'
    }

    # Renomear a coluna "estado" no DataFrame de população
    dados_populacao['state'] = dados_populacao['state'].replace(mapeamento_estados)

    # Ajustar colunas e converter data
    dados_covid_cidades['city'] = dados_covid_cidades['name']
    dados_covid_cidades['date'] = pd.to_datetime(dados_covid_cidades['date'], format='%Y-%m-%d')

    # Merge com dados de população
    resultado = dados_covid_cidades.merge(dados_populacao, on=['city', 'state'], how='inner')

    # Criar taxa de mortalidade
    resultado['death_rate'] = resultado['deaths'] / resultado['population']

    # Criar coluna "confirmed" aleatória para simulação
    np.random.seed(42)
    resultado['confirmed'] = (np.random.lognormal(mean=10, sigma=1.5, size=len(resultado))).astype(int)
    resultado['confirmed'] = resultado['confirmed'].clip(100, 500000)

    # Aplicar uma Window Function: variação percentual de casos nos últimos 7 dias
    resultado['confirmed_pct_change'] = (
        resultado.groupby('city')['confirmed']
        .pct_change(periods=7)
        .fillna(0)
    )

    return resultado


# Aplicar a função
resultado = cidades_de_alto_impacto(dados_covid_cidades, dados_populacao)

resultado2 = transformar_dados_covid(dados_covid_cidades, dados_populacao)

salvar_dataframe_csv(resultado, 'output/cidades_de_alto_impacto.csv')
salvar_dataframe_csv(resultado2, 'output/transformar_dados_covid.csv')

end_time = time.time()
execution_time = end_time - start_time

print(f"Tempo de execução {execution_time:.4f} em segundos para o arquivo: {os.path.basename(__file__)}")