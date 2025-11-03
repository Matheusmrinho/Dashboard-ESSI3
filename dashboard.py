import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Dashboard Inteligente de Testes", layout="wide")

st.title("🧩 Dashboard Inteligente de Casos de Teste")
st.markdown("""
Este painel identifica **problemas estruturais, redundâncias e padrões** nos casos de teste.
""")

# ==============================
# LEITURA DOS ARQUIVOS
# ==============================
st.sidebar.header("📁 Configurações")
data_dir = st.sidebar.text_input("Caminho da pasta com os arquivos CSV", "dados")

if os.path.exists(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if csv_files:
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(os.path.join(data_dir, file), encoding="utf-8", sep=None, engine="python")
            except UnicodeDecodeError:
                df = pd.read_csv(os.path.join(data_dir, file), encoding="latin1", sep=None, engine="python")

            df.columns = [
                "Story Link", "TC ID", "Título do Teste", "Pré-condição",
                "Passos", "Resultado Esperado", "Resultado Execução", "Bug", "Prioridade"
            ]
            df["Arquivo"] = file
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # ==============================
        # MÉTRICAS DERIVADAS
        # ==============================
        data["Qtd Passos"] = data["Passos"].apply(lambda x: len(str(x).split("\n")) if pd.notna(x) else 0)
        data["Tem Pré-condição"] = data["Pré-condição"].notna()
        data["Tem Passos"] = data["Passos"].notna()
        data["Tem Resultado Esperado"] = data["Resultado Esperado"].notna()
        data["Tem Bug"] = data["Bug"].notna()
        data["Tamanho Steps"] = data["Passos"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        data["Tamanho Resultado"] = data["Resultado Esperado"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        # ==============================
        # MÉTRICAS GERAIS
        # ==============================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Testes", len(data))
        col2.metric("Com Pré-condição", data["Tem Pré-condição"].sum())
        col3.metric("Com Bug Reportado", data["Tem Bug"].sum())
        col4.metric("Sem Resultado Esperado", len(data) - data["Tem Resultado Esperado"].sum())

        # ==============================
        # DISTRIBUIÇÕES E GRÁFICOS
        # ==============================
        st.subheader("📈 Estrutura e Complexidade dos Testes")

        fig1 = px.histogram(data, x="Qtd Passos", nbins=20, title="Distribuição da Quantidade de Passos")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(data, y="Tamanho Steps", color="Prioridade", title="Comprimento dos Steps por Prioridade")
        st.plotly_chart(fig2, use_container_width=True)

        # ==============================
        # TESTES EXTREMOS
        # ==============================
        st.subheader("⚠️ Testes com Estrutura Anômala")

        longos = data[data["Qtd Passos"] > data["Qtd Passos"].mean() + 2*data["Qtd Passos"].std()]
        curtos = data[data["Qtd Passos"] < max(1, data["Qtd Passos"].mean() - data["Qtd Passos"].std())]

        if len(longos) > 0:
            st.markdown(f"🔹 **{len(longos)} testes** com passos excessivos (complexos demais).")
            st.dataframe(longos[["TC ID", "Título do Teste", "Qtd Passos", "Prioridade"]])
        else:
            st.info("Nenhum teste excessivamente longo encontrado.")

        if len(curtos) > 0:
            st.markdown(f"🔹 **{len(curtos)} testes** com poucos passos (potencialmente mal definidos).")
            st.dataframe(curtos[["TC ID", "Título do Teste", "Qtd Passos", "Prioridade"]])

        # ==============================
        # DENSIDADE DE BUGS POR PRIORIDADE
        # ==============================
        st.subheader("🐞 Densidade de Bugs por Prioridade")
        bugs_por_prioridade = data.groupby("Prioridade")["Tem Bug"].mean().reset_index()
        bugs_por_prioridade["Tem Bug"] = bugs_por_prioridade["Tem Bug"] * 100
        fig3 = px.bar(bugs_por_prioridade, x="Prioridade", y="Tem Bug",
                      title="Percentual de Testes com Bug por Prioridade", text_auto=".1f")
        st.plotly_chart(fig3, use_container_width=True)

        # ==============================
        # SIMILARIDADE ENTRE TESTES
        # ==============================
        st.subheader("🔍 Testes com Steps Muito Parecidos")

        stopwords_pt = [
            "de", "a", "o", "e", "que", "do", "da", "em", "para", "com", "não", "uma",
            "os", "no", "se", "na", "por", "as", "dos", "como", "mas", "foi", "ao",
            "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", 
            "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
            "ela", "entre", "sem", "mesmo", "me", "esse", "eles", "você", "meu", "minha"
        ]

        tfidf = TfidfVectorizer(stop_words=stopwords_pt)
        tfidf_matrix = tfidf.fit_transform(data["Passos"].fillna(""))

        sim_matrix = cosine_similarity(tfidf_matrix)
        similares = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i, j] > 0.7:
                    similares.append((data.loc[i, "TC ID"], data.loc[j, "TC ID"], round(sim_matrix[i, j], 2)))

        if similares:
            sim_df = pd.DataFrame(similares, columns=["Teste A", "Teste B", "Similaridade"])
            st.dataframe(sim_df, use_container_width=True, height=300)
        else:
            st.info("Nenhum par de testes com alta similaridade (limiar > 0.7).")

        # ==============================
        # RESULTADOS REPETIDOS
        # ==============================
        st.subheader("🔁 Testes com Mesmo Resultado Esperado")

        resultado_repetido = data["Resultado Esperado"].value_counts()
        duplicados = resultado_repetido[resultado_repetido > 1].index.tolist()
        if duplicados:
            rep = data[data["Resultado Esperado"].isin(duplicados)][
                ["TC ID", "Título do Teste", "Resultado Esperado", "Prioridade"]
            ]
            st.dataframe(rep, use_container_width=True, height=300)
        else:
            st.info("Nenhum resultado esperado repetido encontrado.")

    else:
        st.warning("Nenhum arquivo CSV encontrado na pasta especificada.")
else:
    st.info("Informe o caminho da pasta com os arquivos CSV (ex: `dados/`).")
