import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Título do aplicativo
st.title("Predição de Preços de Imóveis em São Paulo")

# Caminho do arquivo CSV
file_path = r"C:\Users\vitor\OneDrive\Documentos\app_imoveis\sao-paulo-properties-april-2019.csv"

# Leitura dos dados
try:
    data = pd.read_csv(file_path)

    # Limpeza e pré-processamento dos dados
    data = data.drop(columns=["Latitude", "Longitude",
                     "Property Type", "New"], errors="ignore")
    label_encoder = LabelEncoder()
    data["District"] = label_encoder.fit_transform(
        data["District"])  # Codificando os distritos

    # Separação por tipo de negociação
    data_rent = data[data["Negotiation Type"] ==
                     "rent"].drop(columns="Negotiation Type")
    data_sale = data[data["Negotiation Type"] ==
                     "sale"].drop(columns="Negotiation Type")

    # Escolha do tipo de análise
    analysis_type = st.selectbox(
        "Selecione o tipo de análise", ["Aluguel", "Venda"])
    data_selected = data_rent if analysis_type == "Aluguel" else data_sale

    # Separação entre características e alvo
    X = data_selected.drop(columns="Price")
    y = data_selected["Price"]

    # Padronização dos dados (sem incluir a coluna "Price")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=12)

    # Treinamento com o melhor modelo (XGBoost)
    st.write(f"Treinando o modelo para {analysis_type}...")
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Avaliação do modelo
    y_pred = model.predict(X_test)

    # Previsão personalizada
    st.write("### Insira os valores do imóvel para prever o preço")
    input_features = {}

    # Para cada coluna, se for "District" usa o selectbox, caso contrário continua com número de entrada ou selectbox
    for col in X.columns:
        if col == "District":
            # Usando selectbox para escolher o distrito
            # Revertendo para nome do bairro
            value = st.selectbox(f"{col}:", options=label_encoder.classes_)
            input_features[col] = label_encoder.transform(
                [value])[0]  # Convertendo para número codificado
        elif col == "Elevator" or col == "Furnished" or col == "Swimming Pool":
            value = st.selectbox(f"{col}:", ["Yes", "No"])
            # Convertendo "Yes" para 1 e "No" para 0
            input_features[col] = 1 if value == "Yes" else 0
        else:
            # Aceita apenas valores inteiros
            value = st.number_input(f"{col}:", value=0, step=1, format="%d")
            input_features[col] = value

    if st.button("Prever Preço"):
        input_df = pd.DataFrame([input_features])
        # Escalonamento dos valores de entrada
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        predicted_price = prediction[0]

        # Reverter o distrito para o nome original
        district_name = input_features["District"]
        district_name = label_encoder.inverse_transform([district_name])[0]

        st.write(f"### O preço estimado do imóvel ({analysis_type}) no bairro {
                 district_name} é: **R$ {predicted_price:,.2f}**")

except FileNotFoundError:
    st.error(f"O arquivo no caminho '{
             file_path}' não foi encontrado. Verifique o caminho e tente novamente.")
