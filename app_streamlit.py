
import streamlit as st
import joblib
import pandas as pd

from preprocessing import DropCols, ImcCalculator, safe_onehot_encoder

model = joblib.load('model_pipeline.joblib')
col_order = pd.read_csv('col_order.csv', header=None)[0].tolist()

st.title('Sistema Preditivo de Obesidade')

st.write(
    'Esta aplicação utiliza um modelo de Machine Learning para estimar '
    'o nível de obesidade com base em informações demográficas e comportamentais.')


# Mapeamento para interface em português → modelo em inglês
gender_map = {
    "Masculino": "Male",
    "Feminino": "Female"
}

faf_labels = {
    0: "Sedentário",
    1: "Baixa",
    2: "Moderada",
    3: "Alta"
}


# Criar widgets dinamicamente com base no col_order
inputs = {}

for col in col_order:
    if col == 'Age':
        inputs[col] = st.slider("Idade", min_value=1, max_value=120, value=30)
    elif col == 'Height':
        inputs[col] = st.slider("Altura (em metros)", min_value=1.0, max_value=2.5, value=1.70, format='%.2f')
    elif col == 'Weight':
        inputs[col] = st.slider("Peso (em quilos)", min_value=10, max_value=300, value=70)
    elif col == "Gender":
        gender_pt = st.selectbox(
            "Gênero",
            options=list(gender_map.keys())
        )
        inputs[col] = gender_map[gender_pt]
    elif col == "FAF":
        faf_value = st.selectbox(
            "Frequência de Atividade Física",
            options=[0, 1, 2, 3],
            format_func=lambda x: faf_labels[x]
        )
        inputs[col] = faf_value
    else:
        inputs[col] = st.text_input(col)


if st.button("🔍 Realizar Predição"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]

    st.success(f"📊 Nível de Obesidade Previsto: **{prediction}**")

    st.warning(
        "⚠️ Este sistema é uma ferramenta de apoio à decisão clínica "
        "e não substitui a avaliação de um profissional de saúde."
    )

    # Verifica se o modelo permite predict_proba
    try:
        proba = model.predict_proba(input_df)
        st.write(pd.DataFrame(proba, columns=model.classes_))
    except:
        pass
