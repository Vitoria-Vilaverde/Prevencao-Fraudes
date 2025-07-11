import streamlit as st
import pandas as pd
import numpy as np
from modelos import treinar_modelo, aplicar_modelo, explicar_predicao
from regras import aplicar_regras, get_regras
from utils import preprocessar_credit, preprocessar_ieee, visualizar_metricas

st.set_page_config(page_title="MVP Prevenção a Fraudes", layout="wide")

st.title("MVP de Prevenção a Fraudes")
st.markdown("Demo integrando duas bases públicas de fraude com pipeline completo: upload, score, regras, métricas, histórico e explicação.")

# Upload/Seleção das Bases
col1, col2 = st.columns(2)
with col1:
    base_selecionada = st.radio("Escolha a base para análise:", 
        options=["Credit Card Fraud (Kaggle)", "IEEE-CIS Fraud Detection", "Unir as duas bases"])
with col2:
    arquivo_manual = st.file_uploader("Ou faça upload do seu próprio .csv", type="csv")

# Leitura e preprocessamento
if base_selecionada == "Credit Card Fraud (Kaggle)":
    df = pd.read_csv("data/creditcard.csv")
    df = preprocessar_credit(df)
elif base_selecionada == "IEEE-CIS Fraud Detection":
    df = pd.read_csv("data/ieee_fraud.csv")
    df = preprocessar_ieee(df)
elif base_selecionada == "Unir as duas bases":
    df1 = pd.read_csv("data/creditcard.csv")
    df2 = pd.read_csv("data/ieee_fraud.csv")
    df1 = preprocessar_credit(df1)
    df2 = preprocessar_ieee(df2)
    df = pd.concat([df1, df2], ignore_index=True)
if arquivo_manual is not None:
    df = pd.read_csv(arquivo_manual)

# Visualização rápida dos dados
st.subheader("Amostra dos Dados")
st.dataframe(df.head())

# Modelagem preditiva
if st.button("Treinar & Aplicar Modelo de Score"):
    modelo, explicador = treinar_modelo(df)
    scores = aplicar_modelo(modelo, df)
    st.session_state['scores'] = scores
    st.success("Modelo treinado e score aplicado!")

    # Visualização de métricas
    visualizar_metricas(df['Class'], scores)

    # Histórico e explicação
    st.subheader("Histórico e Explicação do Score")
    idx = st.slider("Selecione uma transação para explicação", 0, len(df)-1, 0)
    explicacao = explicar_predicao(explicador, df.iloc[idx])
    st.json(explicacao)

# Simulação de regras
st.subheader("Simulação de Regras de Negócio")
regras = get_regras()
st.write("Regras atuais:")
st.json(regras)

if st.button("Aplicar Motor de Regras"):
    df['flag_regra'] = df.apply(aplicar_regras, axis=1)
    st.write(df[['flag_regra']].value_counts())
    st.success("Regras aplicadas!")

    # Métricas da regra vs modelo
    if 'scores' in st.session_state:
        st.subheader("Comparação Modelo x Regras")
        st.write("Você pode analisar os falsos positivos, falsos negativos, acurácia de cada abordagem.")

# Download de resultados
if st.button("Baixar resultados com scores e flags"):
    df['score_modelo'] = st.session_state.get('scores', np.nan)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "resultados_fraude.csv")

