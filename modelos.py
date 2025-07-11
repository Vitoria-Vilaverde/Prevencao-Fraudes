import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap

def treinar_modelo(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer

def aplicar_modelo(model, df):
    X = df.drop(columns=['Class'])
    return model.predict_proba(X)[:,1]

def explicar_predicao(explainer, row):
    shap_values = explainer.shap_values(row)
    top_features = sorted(zip(row.index, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
    return {k: float(v) for k,v in top_features}
