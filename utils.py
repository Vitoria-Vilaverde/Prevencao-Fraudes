def preprocessar_credit(df):
    # Exemplo: Remover coluna 'Time', garantir colunas mínimas
    df = df.copy()
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    return df

def preprocessar_ieee(df):
    # Exemplo: manter apenas colunas que existem também no creditcard.csv
    min_cols = ['TransactionAmt', 'isFraud']
    rename_dict = {'TransactionAmt': 'Amount', 'isFraud': 'Class'}
    for old, new in rename_dict.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    # Se quiser selecionar só as principais
    keep = [col for col in df.columns if col in ['Amount', 'Class']]
    df = df[keep]
    return df

from sklearn.metrics import roc_auc_score, f1_score, classification_report
import streamlit as st

def visualizar_metricas(y_true, y_pred):
    st.write("AUC:", roc_auc_score(y_true, y_pred))
    st.write("F1 Score:", f1_score(y_true, y_pred > 0.5))
    st.text(classification_report(y_true, y_pred > 0.5))
