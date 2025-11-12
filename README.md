# projeto-medalhao
import os
import json
import requests
import pandas as pd
import time

# ===============================================================
#  ETAPA 1 - EXTRAÇÃO E CAMADA BRONZE
# ===============================================================

API_URL = "https://brasil.io/api/v1/dataset/gastos-diretos/gastos/data/"
API_KEY = "dbfc69a93c12ae9958b1e37b72bab280ca8ae1bb"

RAW_PATH = "dataset/raw"
BRONZE_PATH = "dataset/bronze"

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(BRONZE_PATH, exist_ok=True)

# Função: baixar dados da API Brasil.IO e salvar em JSON
def extrair_dados(pagina: int = 1):
    """Baixa os dados da API Brasil.io e salva na pasta RAW."""
    url = f"{API_URL}?page={pagina}"
    resposta = requests.get(url, headers={"Authorization": f"Token {API_KEY}"})

    if resposta.status_code != 200:
        print(f"Erro ao acessar página {pagina}: {resposta.status_code}")
        return False

    dados = resposta.json()

    if "results" not in dados or not dados["results"]:
        return False

    with open(f"{RAW_PATH}/gastos_{pagina}.json", "w", encoding="utf-8") as f:
        json.dump(dados["results"], f, indent=4, ensure_ascii=False)

    print(f"Página {pagina} salva em {RAW_PATH}")
    return True

# Função: transformar JSONs da RAW em Parquet (camada Bronze)
def transformar_dados():
    """Lê todos os JSONs da pasta RAW e salva como Parquet na BRONZE."""
    arquivos = [f for f in os.listdir(RAW_PATH) if f.endswith(".json")]
    todos_dados = []

    for arquivo in arquivos:
        with open(os.path.join(RAW_PATH, arquivo), "r", encoding="utf-8") as f:
            dados = json.load(f)
            todos_dados.extend(dados)

    df = pd.DataFrame(todos_dados)

    if "ano" not in df.columns or "mes" not in df.columns:
        print("Dados não possuem colunas 'ano' e 'mes'. Salvando sem particionar.")
        df.to_parquet(f"{BRONZE_PATH}/gastos.parquet", index=False)
    else:
        df.to_parquet(
            BRONZE_PATH,
            engine="pyarrow",
            partition_cols=["ano", "mes"],
            index=True,
        )

    print(f"Dados salvos em formato Parquet na pasta {BRONZE_PATH}")

# Pipeline completa (extração + transformação)
def pipeline():
    """Executa a extração de várias páginas e depois a transformação."""
    pagina = 1
    max_paginas = 100  # limite para evitar excesso de requisições

    while pagina <= max_paginas and extrair_dados(pagina):
        time.sleep(1.5)
        pagina += 1

    if pagina > max_paginas:
        print(f"Limite de {max_paginas} páginas atingido.")

    transformar_dados()
    print("Processo Bronze concluído com sucesso!")


# ===============================================================
#  EXECUÇÃO CONTROLADA (só baixa se ainda não tiver dados)
# ===============================================================

if __name__ == "__main__":
    # Só executa a extração se a pasta RAW estiver vazia
    if not os.listdir(RAW_PATH):
        print("Nenhum dado encontrado na pasta RAW. Iniciando extração da API...")
        pipeline()
    else:
        print("Dados já existem na pasta RAW. Pulando extração...")


# ===============================================================
#  ETAPA 2 - LEITURA DA CAMADA BRONZE
# ===============================================================

print("\nLendo dados da camada Bronze...")
BRONZE_PATH = "dataset/bronze"
arquivo_parquet = os.path.join(BRONZE_PATH, "gastos.parquet")

if os.path.exists(arquivo_parquet):
    df = pd.read_parquet(arquivo_parquet)
else:
    df = pd.concat(
        [pd.read_parquet(os.path.join(root, f))
         for root, dirs, files in os.walk(BRONZE_PATH)
         for f in files if f.endswith(".parquet")],
        ignore_index=True
    )

print("Dados carregados com sucesso!")
print(df.info())
print(df.head())


# ===============================================================
#  ETAPA 3 - CAMADA SILVER (LIMPEZA E PADRONIZAÇÃO)
# ===============================================================

SILVER_PATH = "dataset/silver"
os.makedirs(SILVER_PATH, exist_ok=True)

# 1. Limpeza
def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e padroniza os dados da camada Bronze."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Converter 'valor' para número
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    # Identificar e tratar coluna de data
    coluna_data = None
    for c in df.columns:
        if "data" in c:
            coluna_data = c
            break

    if coluna_data:
        df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
        df["ano"] = df[coluna_data].dt.year
        df["mes"] = df[coluna_data].dt.month
        print(f"Coluna de data usada: {coluna_data}")
    else:
        print("Nenhuma coluna de data encontrada — sem partições de ano/mês.")

    # Preencher valores nulos
    for col in ["orgao", "favorecido", "nome_orgao", "nome_favorecido"]:
        if col in df.columns:
            df[col] = df[col].fillna("Não informado")

    # Remover duplicatas
    df = df.drop_duplicates()

    print("Dados limpos e padronizados!")
    return df


# 2. Testes simples de qualidade
def testes_qualidade(df: pd.DataFrame):
    print("\nTestes de qualidade dos dados:")
    print(f"Total de linhas: {len(df)}")
    if "valor" in df.columns:
        print(f"Valores nulos em 'valor': {df['valor'].isna().sum()}")
    if "orgao" in df.columns:
        print(f"Valores nulos em 'orgao': {df['orgao'].isna().sum()}")
    print("Testes concluídos.\n")


# 3. Salvamento da camada Silver
def salvar_silver(df: pd.DataFrame):
    if "ano" in df.columns and "mes" in df.columns:
        df.to_parquet(
            SILVER_PATH,
            engine="pyarrow",
            partition_cols=["ano", "mes"],
            index=False
        )
        print("Dados salvos com partições por ano/mês.")
    else:
        df.to_parquet(f"{SILVER_PATH}/gastos_silver.parquet", index=False)
        print("Dados salvos sem particionamento (ano/mês não encontrados).")
    print(f"Arquivos salvos na camada Silver: {SILVER_PATH}")


# 4. Execução Silver
print("\nIniciando limpeza e transformação (camada Silver)...")
df_silver = limpar_dados(df)
testes_qualidade(df_silver)
salvar_silver(df_silver)
print("Etapa Silver concluída com sucesso!")
