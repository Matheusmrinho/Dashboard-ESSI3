import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Dashboard Inteligente de Testes", layout="wide")

st.title("🧩 Dashboard Inteligente de Casos de Teste")
st.markdown("""
Este painel identifica **problemas estruturais, redundâncias e padrões** nos casos de teste.
Agora também diferencia **Testes Unitários vs Testes de Integração** automaticamente.
""")

# ==============================
# LEITURA E EXTRAÇÃO DA US
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

            # Padroniza colunas
            df.columns = [
                "Story Link", "TC ID", "Título do Teste", "Pré-condição",
                "Passos", "Resultado Esperado", "Resultado Execução",
                "Bug", "Prioridade"
            ]

            df["Arquivo"] = file
            # Extrai US do nome do arquivo
            us = file.split("US")[1].split(".")[0] if "US" in file else "Desconhecida"
            df["US"] = f"US{us}"
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)


        # ==============================
        # FILTRO POR US
        # ==============================
        st.sidebar.subheader("Filtro por User Story")
        us_list = sorted(data["US"].unique())
        us_selecionadas = st.sidebar.multiselect("Selecione US", us_list, default=us_list)

        if us_selecionadas:
            data = data[data["US"].isin(us_selecionadas)]

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
            data["Bug_Resolvido"] = (
            data["Resultado Execução"].str.contains("PASSED", case=False, na=False) &
            data["Bug"].notna() &
            (data["Bug"].astype(str).str.strip() != "")
            )
            resolvidos = data["Bug_Resolvido"].sum()

            # ==============================
            # STATUS DOS TESTES
            # ==============================
            st.subheader("📊 Status dos Testes")
            # ==============================
            # CONTAGEM GERAL
            # ==============================
            total_testes = len(data)
            passaram = data["Resultado Execução"].str.contains("PASSED|OK|SUCESS", case=False, na=False).sum()
            falharam = data["Resultado Execução"].str.contains("FAILED|ERRO|FAILED", case=False, na=False).sum()
            nao_exec = data["Resultado Execução"].str.contains("NÃO EXECUTADO|NAO EXECUTADO", case=False, na=False).sum()

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total de Testes", total_testes)
            col2.metric("Passaram", passaram)
            col3.metric("Falharam", falharam)
            col4.metric("Não Executados", nao_exec)
            col5.metric("Com Bug", data["Tem Bug"].sum())
            col6.metric("Bugs Resolvidos", resolvidos)

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
            # ==============================
            # ⚠️ TESTES QUE BLOQUEIAM ENTREGA
            # ==============================
            st.subheader("🚫 Risco de Entrega – Testes NÃO Executados de Alta Prioridade")
            bloqueio = data[(data["Prioridade"] == "ALTA") &
                            (data["Resultado Execução"].str.contains("NÃO EXECUTADO", case=False, na=False))]
            if not bloqueio.empty:
                st.warning(f"{len(bloqueio)} testes-críticos ainda não foram executados.")
                st.dataframe(bloqueio[["US", "TC ID", "Título do Teste", "Prioridade"]])
            else:
                st.success("Todos os testes de alta prioridade já foram executados.")


            # ==============================
            # 📈 EVOLUÇÃO DE QUALIDADE POR SPRINT (US)
            # ==============================
            st.subheader("📊 Qualidade da Entrega – Taxa de Sucesso por US")
            evo = data.groupby("US").agg(
                Total=("TC ID", "count"),
                Passados=("Resultado Execução", lambda s: s.str.contains("PASSED", case=False).sum())
            ).reset_index()
            evo["Taxa de Sucesso (%)"] = (evo["Passados"] / evo["Total"] * 100).round(1)
            fig_evo = px.bar(evo, x="US", y="Taxa de Sucesso (%)", text="Taxa de Sucesso (%)",
                            title="Taxa de Sucesso por User Story",
                            color="Taxa de Sucesso (%)", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_evo, use_container_width=True)

            
            # ==============================
            # RESULTADOS POR US (COM NÃO EXECUTADO)
            # ==============================
            res_us = data.groupby(["US", "Resultado Execução"]).size().unstack(fill_value=0)

            # garante as três colunas, mesmo que vazias
            for col in ["PASSED", "FAILED", "NÃO EXECUTADO"]:
                if col not in res_us:
                    res_us[col] = 0
            res_us = res_us[["PASSED", "FAILED", "NÃO EXECUTADO"]]

            fig_us = px.bar(res_us, x=res_us.index, y=["PASSED", "FAILED", "NÃO EXECUTADO"],
                            title="Resultados por User Story", barmode="group",
                            color_discrete_map={"PASSED": "green",
                                                "FAILED": "red",
                                                "NÃO EXECUTADO": "gray"})
            st.plotly_chart(fig_us, use_container_width=True)

           # ==========================================================
            # 📈 Taxa de Resolução de Bugs por US
            # ==========================================================
            # 1) cria coluna booleana: FAILED que tem link
            data["FAILED_com_link"] = (
                data["Resultado Execução"].str.contains("FAILED", case=False, na=False) &
                data["Bug"].notna() &
                (data["Bug"].astype(str).str.strip() != "")
            )

            # 2) agrega por US
            taxa_df = (
                data.groupby("US")
                .agg(
                    Total_Bugs=("FAILED_com_link", "sum"),
                    Resolvidos=("Bug_Resolvido", "sum")
                )
                .assign(Taxa=lambda df: (df["Resolvidos"] / df["Total_Bugs"] * 100).round(1))
                .fillna(0)
                .reset_index()
            )

            st.subheader("📈 Taxa de Resolução de Bugs")
            st.dataframe(taxa_df, use_container_width=True)
            # ==============================
            # TESTES CRÍTICOS FALHADOS
            # ==============================
            st.subheader("🚨 Testes Críticos com Falha")
            criticos = data[(data["Prioridade"] == "ALTA") & (data["Resultado Execução"].str.contains("FAILED", case=False, na=False))]
            if not criticos.empty:
                st.dataframe(criticos[["US", "TC ID", "Título do Teste", "Bug", "Prioridade"]])
            else:
                st.success("Nenhum teste crítico falhou.")

            # ==============================
            # SUMÁRIO POR US
            # ==============================
            st.subheader("📊 Sumário por User Story")
            sumario_us = data.groupby("US").agg(
                Total=("TC ID", "count"),
                Passaram=("Resultado Execução", lambda x: x.str.contains("PASSED", case=False).sum()),
                Falharam=("Resultado Execução", lambda x: x.str.contains("FAILED", case=False).sum()),
                Com_Bug=("Tem Bug", "sum")
            ).reset_index()
            sumario_us["Taxa de Sucesso (%)"] = (sumario_us["Passaram"] / sumario_us["Total"] * 100).round(1)
            st.dataframe(sumario_us)

        

            # ==============================
            #💚 Caixa de Luz Verde – US 100 % Aprovadas
            # ==============================
            luz_verde = sumario_us[sumario_us["Taxa de Sucesso (%)"] == 100]
            if not luz_verde.empty:
                fig_lv = px.bar(luz_verde.sort_values("Total"), y="US", x="Total",
                                orientation="h", text="Total",
                                title="💚 US 100 % Aprovadas – Prontas para Entrega",
                                color="Total", color_continuous_scale="Greens")
                fig_lv.update_traces(texttemplate="%{x} casos", textposition="outside")
                st.plotly_chart(fig_lv, use_container_width=True)
            else:
                st.info("Nenhuma US está 100 % aprovada ainda.")

            # ==============================
            # SIMILARIDADE ENTRE TESTES
            # ==============================
            st.subheader("🔍 Testes com Steps Muito Parecidos")
            stopwords_pt = [
                "de", "a", "o", "e", "que", "do", "da", "em", "para", "com", "não", "uma", "os", "no", "se",
                "na", "por", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua",
                "ou", "ser", "quando", "muito", "nos", "já", "está", "eu", "também", "só", "pelo", "pela",
                "até", "isso", "ela", "entre", "sem", "mesmo", "me", "esse", "eles", "você", "meu", "minha"
            ]

            data = data.reset_index(drop=True)
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
                st.dataframe(sim_df, height=300)
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
                    ["US", "TC ID", "Título do Teste", "Resultado Esperado", "Prioridade"]
                ]
                st.dataframe(rep, height=300)
            else:
                st.info("Nenhum resultado esperado repetido encontrado.")

        else:
            st.warning("Nenhuma US selecionada.")
    else:
        st.warning("Nenhum arquivo CSV encontrado na pasta especificada.")
else:
    st.info("Informe o caminho da pasta com os arquivos CSV (ex: `dados/`).")