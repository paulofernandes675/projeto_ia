import requests
import pandas as pd
import datetime
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Função para consultar dados de temperatura diária do OpenWeather para Lisboa
def consultar_temperaturas(api_key, cidade="Lisboa"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={cidade}&appid={api_key}&units=metric&lang=pt_br"
        response = requests.get(url)
        response.raise_for_status()

        dados = response.json()

        if "list" not in dados:
            st.error("Dados não encontrados na resposta JSON.")
            return None

        previsoes = []
        for item in dados["list"]:
            previsoes.append({
                "data": item["dt_txt"],
                "temp_max": item["main"]["temp_max"],
                "temp_min": item["main"]["temp_min"],
                "umidade": item["main"]["humidity"],
                "vento": item["wind"]["speed"]
            })

        return pd.DataFrame(previsoes)

    except requests.exceptions.RequestException as e:
        st.error(f"Erro de requisição: {e}")
    except ValueError as e:
        st.error(f"Erro de processamento de dados: {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
    return None

# Função para treinar e implementar o modelo
def treinar_modelo(dados):
    try:
        if dados is None or dados.empty:
            st.error("Sem dados para treinar o modelo.")
            return None

        dados["data"] = pd.to_datetime(dados["data"])
        dados["dia_do_ano"] = dados["data"].dt.dayofyear

        X = dados[["dia_do_ano", "umidade", "vento"]]
        y = dados["temp_max"]

        modelo = LinearRegression()
        modelo.fit(X, y)

        return modelo

    except Exception as e:
        st.error(f"Erro ao treinar o modelo: {e}")
        return None

# Função principal
def main():
    api_key = "9fb3b149f965426caa9d207c420edc19"  # Substitua pela sua chave de API
    cidade = "Lisboa"

    st.title("Previsão de Temperatura - Lisboa")

    # Coleta os dados
    dados = consultar_temperaturas(api_key, cidade)

    if dados is not None:
        st.subheader("Dados Coletados:")
        st.dataframe(dados.head())  # Exibe os primeiros dados coletados

        modelo = treinar_modelo(dados)

        if modelo is not None:
            st.subheader("Previsões para os Próximos Dias:")
            previsoes = []

            # Faz previsões para os próximos dias
            novos_dados = pd.DataFrame({
                "dia_do_ano": [datetime.datetime.now().timetuple().tm_yday + i for i in range(1, 6)],
                "umidade": [70, 65, 75, 80, 60],  # Exemplo de valores fictícios de umidade
                "vento": [5, 10, 8, 6, 7]  # Exemplo de valores fictícios de velocidade do vento
            })

            previsoes = modelo.predict(novos_dados)

            for i, temp in enumerate(previsoes):
                st.write(f"Dia {i + 1}: Temperatura máxima prevista: {temp:.2f}°C")

# Executa a aplicação Streamlit
if __name__ == "__main__":
    main()
