import polars as pl
from timeit_decorator import timeit_decorator
import os
import time
import numpy as np

start_time = time.time()

@timeit_decorator
def ler_dataframe_csv(file):
    schema_overrides = {
        'health_region_code': pl.Utf8 
    }
    df = pl.read_csv(file, schema_overrides=schema_overrides)
    return df

@timeit_decorator
def salvar_dataframe_csv(df, file):
    df.write_csv(file)

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
    dados_populacao = dados_populacao.with_columns(
        replaced=pl.col('state').replace(mapeamento_estados)
    )
    
    dados_covid_cidades = dados_covid_cidades.with_columns(
        pl.col('name').alias('city')
    )
    
    # Converter a coluna "data" para o formato datetime
    dados_covid_cidades = dados_covid_cidades.with_columns(
        pl.col('date').str.to_datetime(format='%S%M%H%d%m%Y',strict=False)
    )
    
    # Realizar o join com os dados de população
    resultado = dados_covid_cidades.join(dados_populacao, on=['city', 'state'], how='inner')
    
    # Adicionar uma coluna de taxa de mortalidade
    resultado = resultado.with_columns(
        (pl.col('deaths') / pl.col('population')).alias('death_rate')
    )
    
    # Calcular a média móvel de 7 dias para óbitos
    resultado = resultado.with_columns(
        pl.col('deaths').rolling_mean(window_size=7).over('state').alias('7_day_avg_deaths')
    )

    # Ordenar pelos valores de taxa de mortalidade em ordem decrescente
    resultado = resultado.sort('death_rate', descending=True)
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
    dados_populacao = dados_populacao.with_columns(
        pl.col('state').replace(mapeamento_estados).alias("state")
    )

    # Renomear a coluna 'name' para 'city'
    dados_covid_cidades = dados_covid_cidades.with_columns(
        pl.col('name').alias('city')
    )

    # Converter a coluna "data" para o formato datetime
    dados_covid_cidades = dados_covid_cidades.with_columns(
        pl.col('date').str.to_datetime(format='%S%M%H%d%m%Y', strict=False)
    )

    # Realizar o join com os dados de população
    resultado = dados_covid_cidades.join(dados_populacao, on=['city', 'state'], how='inner')

    # Adicionar uma coluna de taxa de mortalidade
    resultado = resultado.with_columns(
        (pl.col('deaths') / pl.col('population')).alias('death_rate')
    )

    # Criar uma coluna "confirmed" aleatória para simulação
    np.random.seed(42)
    confirmed_random = np.random.lognormal(mean=10, sigma=1.5, size=resultado.height).astype(int)
    confirmed_random = np.clip(confirmed_random, 100, 500000)

    resultado = resultado.with_columns(
        pl.Series("confirmed", confirmed_random)
    )

    # Adicionar a transformação de variação percentual dos casos confirmados
    resultado = resultado.with_columns(
        ((pl.col("confirmed") / pl.col("confirmed").shift(7)) - 1)
        .fill_null(0)
        .alias("confirmed_pct_change")
    )

    # Ordenar pelos valores de taxa de mortalidade em ordem decrescente
    resultado = resultado.sort('death_rate', descending=True)

    return resultado

# Aplicar a função
resultado = cidades_de_alto_impacto(dados_covid_cidades, dados_populacao)
resultado2 = transformar_dados_covid(dados_covid_cidades, dados_populacao)

salvar_dataframe_csv(resultado, 'output/cidades_de_alto_impacto.csv')
salvar_dataframe_csv(resultado, 'output/transformar_dados_covid.csv')

end_time = time.time()
execution_time = end_time - start_time

print(f"Tempo de execução {execution_time:.4f} em segundos para o arquivo: {os.path.basename(__file__)}")