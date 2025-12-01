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
                "Passos", "Resultado Esperado", "Resultado Execução",
                "Bug", "Prioridade"
            ]

            df["Arquivo"] = file

            # ==============================
            # DETECTA TIPO DE TESTE
            # unitário → final USxxx.csv
            # integração → qualquer outro nome
            # ==============================
            if file.lower().split(".csv")[0].endswith(tuple([f"us{str(i).zfill(3)}" for i in range(1,999)])):
                df["Tipo"] = "Unitário"
            else:
                df["Tipo"] = "Integração"

            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)


        # ==============================
        # TRATAMENTO E LIMPEZA DE DADOS
        # ==============================

        # NOVO: Limpa o nome dos arquivos para exibição na sidebar
        prefixo_comum = "[entrenos] Execução de testes 2025.1 -"
        data["Arquivo Limpo"] = data["Arquivo"].str.replace(prefixo_comum, "", regex=False).str.strip()

        # ==============================
        # FILTRO POR TIPO
        # ==============================
        st.sidebar.subheader("Tipos de Teste")
        tipos = st.sidebar.multiselect(
            "Filtrar por tipo",
            ["Unitário", "Integração"],
            default=["Unitário", "Integração"]
        )
        data = data[data["Tipo"].isin(tipos)]

        # ==============================
        # FILTRO POR ARQUIVO CSV (USANDO O NOME LIMPO)
        # ==============================
        st.sidebar.subheader("Filtro por User Story")

        # 1. Lista de nomes de arquivos limpos únicos
        arquivos_limpos_list = sorted(data["Arquivo Limpo"].dropna().unique().tolist())

        # 2. Multiselect para selecionar os arquivos (agora com nomes curtos)
        arquivos_selecionados_limpos = st.sidebar.multiselect(
            "Selecione o Arquivo CSV", 
            arquivos_limpos_list, 
            default=arquivos_limpos_list
        )

        # 3. Aplica o filtro usando a coluna "Arquivo Limpo"
        if len(arquivos_selecionados_limpos) > 0:
            data = data[data["Arquivo Limpo"].isin(arquivos_selecionados_limpos)]
        else:
            # Se o usuário desmarcar todos, não exibe nada
            data = data.head(0)
            st.warning("Selecione pelo menos um arquivo para visualizar os dados.")

        if "Story Link" in data.columns:
            data = data.rename(columns={"Story Link": "US"})
            
            # 💡 CORREÇÃO 2: Normaliza o nome da US (MUN-xxx para US-xxx)
            data["US"] = data["US"].str.replace("MUN-", "US-", case=False, regex=False)
            
            # Substitui nulos ou espaços vazios por 'N/A' antes de agrupar
            data["US"] = data["US"].replace('', pd.NA).fillna("Integração")
        else:
            # Cria uma coluna US para evitar quebra no código se a coluna Story Link não existir
            data["US"] = "Integração"   

        # ==============================
        # MÉTRICAS DERIVADAS
        # ==============================
        data["Qtd Passos"] = data["Passos"].apply(lambda x: len(str(x).split("\n")) if pd.notna(x) else 0)
        data["Tem Pré-condição"] = data["Pré-condição"].notna()
        data["Tem Passos"] = data["Passos"].notna()
        data["Tem_Bug_Reportado"] = data["Bug"].notna() & (data["Bug"].astype(str).str.strip() != "")
        data["Tem Resultado Esperado"] = data["Resultado Esperado"].notna()
        data["Tem Bug"] = data["Tem_Bug_Reportado"]
        data["Tamanho Steps"] = data["Passos"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        data["Tamanho Resultado"] = data["Resultado Esperado"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        data["Bug_Verificado_Resolvido"] = (
        data["Resultado Execução"].str.contains("PASSED|OK|SUCESS", case=False, na=False) &
        data["Tem_Bug_Reportado"]
        )
        resolvidos_simples = data["Tem_Bug_Reportado"].sum() # <-- Use esta se for seu critério

        # Opção B (Mais robusta - Apenas bugs que PASSARAM e tinham um link):
        resolvidos_verificados = data["Bug_Verificado_Resolvido"].sum()

        # ==============================
        # MÉTRICAS GERAIS POR TIPO
        # ==============================
        st.subheader("📊 Métricas por Tipo de Teste")

        for tipo in ["Unitário", "Integração"]:
            subset = data[data["Tipo"] == tipo]
            if len(subset) == 0:
                continue

            st.markdown(f"## 🔹 {tipo}")
            if tipo == 'Unitário':
                col1, col2, col3, col4, col5,col6 = st.columns(6)
                total = len(subset)
                passaram = subset["Resultado Execução"].str.contains("PASSED|OK|SUCESS", case=False, na=False).sum()
                falharam = subset["Resultado Execução"].str.contains("FAILED|ERRO", case=False, na=False).sum()
                bugs = subset["Tem Bug"].sum()

                col1.metric("Total", total)
                col2.metric("Passaram", passaram)
                col3.metric("Falharam", falharam)
                col4.metric("Bugs", bugs)
                col5.metric("Sem Resultado Esperado", len(subset) - subset["Tem Resultado Esperado"].sum())
                col6.metric("Bugs Verificados Resolvidos", resolvidos_verificados)
            else:
                col1, col2, col3, col4, col5 = st.columns(5)
                total = len(subset)
                passaram = subset["Resultado Execução"].str.contains("PASSED|OK|SUCESS", case=False, na=False).sum()
                falharam = subset["Resultado Execução"].str.contains("FAILED|ERRO", case=False, na=False).sum()
                bugs = subset["Tem Bug"].sum()

                col1.metric("Total", total)
                col2.metric("Passaram", passaram)
                col3.metric("Falharam", falharam)
                col4.metric("Bugs", bugs)
                col5.metric("Sem Resultado Esperado", len(subset) - subset["Tem Resultado Esperado"].sum())
                


            st.divider()

        # ==============================
        # GRÁFICOS DA ESTRUTURA
        # ==============================
        st.subheader("📈 Estrutura e Complexidade dos Testes")

        fig1 = px.histogram(data, x="Qtd Passos", color="Tipo",
                            nbins=20, title="Distribuição da Quantidade de Passos por Tipo")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(data, y="Tamanho Steps", color="Tipo", title="Comprimento dos Steps por Tipo")
        st.plotly_chart(fig2, use_container_width=True)

        # ==============================
        # TESTES EXTREMOS
        # ==============================
        st.subheader("⚠️ Testes com Estrutura Anômala")

        longos = data[data["Qtd Passos"] > data["Qtd Passos"].mean() + 2 * data["Qtd Passos"].std()]
        curtos = data[data["Qtd Passos"] < max(1, data["Qtd Passos"].mean() - data["Qtd Passos"].std())]

        if len(longos) > 0:
            st.markdown(f"🔹 **{len(longos)} testes** com passos excessivos.")
            st.dataframe(longos[["Tipo", "TC ID", "Título do Teste", "Qtd Passos", "Prioridade"]])
        else:
            st.info("Nenhum teste excessivamente longo encontrado.")

        if len(curtos) > 0:
            st.markdown(f"🔹 **{len(curtos)} testes** com poucos passos.")
            st.dataframe(curtos[["Tipo", "TC ID", "Título do Teste", "Qtd Passos", "Prioridade"]])

        # ==============================
        # BUGS POR PRIORIDADE
        # ==============================
        st.subheader("🐞 Densidade de Bugs por Prioridade")
        bugs_por_prioridade = data.groupby("Prioridade")["Tem Bug"].mean().reset_index()
        bugs_por_prioridade["Tem Bug"] *= 100
        fig3 = px.bar(bugs_por_prioridade, x="Prioridade", y="Tem Bug",
                      title="Percentual de Testes com Bug por Prioridade", text_auto=".1f")
        st.plotly_chart(fig3, use_container_width=True)

        # ==============================
        # SIMILARIDADE ENTRE TESTES
        # ==============================
        st.subheader("🔍 Testes com Steps Muito Parecidos")

        data = data.reset_index(drop=True)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(data["Passos"].fillna(""))
        sim_matrix = cosine_similarity(tfidf_matrix)

        similares = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i, j] > 0.7:
                    similares.append((data.loc[i, "TC ID"], data.loc[j, "TC ID"], sim_matrix[i, j]))

        if similares:
            sim_df = pd.DataFrame(similares, columns=["Teste A", "Teste B", "Similaridade"])
            st.dataframe(sim_df, height=300)
        else:
            st.info("Nenhum par de testes altamente similar encontrado.")

        if data["US"].nunique() > 1: # Só faz sentido se houver mais de uma US
    
            st.header("🎯 Análise de Qualidade por User Story")
            
            # 📈 EVOLUÇÃO DE QUALIDADE POR SPRINT (US) - Taxa de Sucesso
            st.subheader("📊 Qualidade da Entrega – Taxa de Sucesso por US")
            evo = data.groupby("US").agg(
                Total=("TC ID", "count"),
                Passados=("Resultado Execução", lambda s: s.str.contains("PASSED|OK|SUCESS", case=False).sum())
            ).reset_index()
            
            # Evita divisão por zero
            evo["Taxa de Sucesso (%)"] = (evo["Passados"] / evo["Total"].replace(0, 1) * 100).round(1)
            
            fig_evo = px.bar(evo, x="US", y="Taxa de Sucesso (%)", text="Taxa de Sucesso (%)",
                            title="Taxa de Sucesso por User Story",
                            color="Taxa de Sucesso (%)", color_continuous_scale="RdYlGn")
            fig_evo.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_evo, use_container_width=True)

            # ---------------------------------------------------------

            # RESULTADOS POR US (COM NÃO EXECUTADO)
            st.subheader("📉 Distribuição de Resultados por US")
            
            # Classifica os resultados em três categorias principais para o gráfico
            data["Status_Agrupado"] = data["Resultado Execução"].apply(lambda x: 
                "PASSED" if str(x).upper() in ["PASSED", "OK", "SUCESS"] else
                "FAILED" if str(x).upper() in ["FAILED", "ERRO", "FALHA"] else
                "NÃO EXECUTADO")
                
            res_us = data.groupby(["US", "Status_Agrupado"]).size().unstack(fill_value=0)
            
            # Garante a ordem das colunas
            colunas_status = [c for c in ["PASSED", "FAILED", "NÃO EXECUTADO"] if c in res_us.columns]
            if colunas_status:
                res_us = res_us[colunas_status]

                fig_us = px.bar(res_us, x=res_us.index, y=colunas_status,
                                title="Resultados por User Story", barmode="group",
                                color_discrete_map={"PASSED": "green",
                                                    "FAILED": "red",
                                                    "NÃO EXECUTADO": "gray"})
                fig_us.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_us, use_container_width=True)
            else:
                st.info("Não há resultados de execução suficientes para o gráfico por US.")

            # ---------------------------------------------------------

            # 📈 Taxa de Resolução de Bugs por US
            # 1) cria coluna booleana: FAILED que tem link (Bugs reportados)
            data["Bugs_Reportados"] = (
                data["Resultado Execução"].str.contains("FAILED|ERRO|FALHA", case=False, na=False) &
                data["Bug"].notna() &
                (data["Bug"].astype(str).str.strip() != "")
            )

            # 2) agrega por US
            taxa_df = (
            data.groupby("US")
            .agg(
                # Contagem de bugs que FALHARAM E TINHAM LINK DE BUG
                Bugs_Reportados=("Bugs_Reportados", "sum"), 
                # Contagem de bugs que PASSARAM E TINHAM LINK DE BUG
                Bugs_Verificados=("Bug_Verificado_Resolvido", "sum") 
            )
            .assign(
                Taxa_Resolucao=lambda df: (
                    # 💡 CORREÇÃO 1: Limita a taxa a 100%
                    (df["Bugs_Verificados"] / df["Bugs_Reportados"].replace(0, 1))
                    .clip(upper=1.0) * 100
                ).round(1)
            )
            .fillna(0)
            .reset_index()
            )

            st.subheader("📈 Taxa de Resolução de Bugs por US")
            # Filtra apenas USs onde bugs foram reportados
            taxa_exibir = taxa_df[taxa_df["Bugs_Reportados"] > 0]
            if not taxa_exibir.empty:
                st.dataframe(taxa_exibir.rename(columns={"Taxa_Resolucao": "Taxa de Resolução (%)"}), use_container_width=True)
            else:
                st.info("Nenhum bug foi reportado ou resolvido para calcular a taxa por US.")

            # ---------------------------------------------------------

            # TESTES CRÍTICOS FALHADOS
            st.subheader("🚨 Testes Críticos com Falha")
            criticos = data[(data["Prioridade"].astype(str).str.upper() == "ALTA") & 
                            (data["Status_Agrupado"] == "FAILED")]
            if not criticos.empty:
                st.dataframe(criticos[["US", "TC ID", "Título do Teste", "Bug", "Prioridade", "Tipo"]], use_container_width=True)
            else:
                st.success("Nenhum teste crítico falhou.")

            # ---------------------------------------------------------
            
            # SUMÁRIO POR US
            st.subheader("📊 Sumário Completo por User Story")
            sumario_us = data.groupby("US").agg(
                Total=("TC ID", "count"),
                Passaram=("Status_Agrupado", lambda x: (x == "PASSED").sum()),
                Falharam=("Status_Agrupado", lambda x: (x == "FAILED").sum()),
                Nao_Executados=("Status_Agrupado", lambda x: (x == "NÃO EXECUTADO").sum()),
                Com_Bug=("Tem Bug", "sum")
            ).reset_index()
            sumario_us["Taxa de Sucesso (%)"] = (sumario_us["Passaram"] / sumario_us["Total"].replace(0, 1) * 100).round(1)
            
            st.dataframe(sumario_us.rename(columns={"Nao_Executados": "Não Executados", "Com_Bug": "Bugs Reportados"}), use_container_width=True)

            # ---------------------------------------------------------

            # 💚 Caixa de Luz Verde – US 100 % Aprovadas
            st.subheader("✅ Status de Entrega")
            luz_verde = sumario_us[sumario_us["Taxa de Sucesso (%)"] == 100]
            
            if not luz_verde.empty:
                fig_lv = px.bar(luz_verde.sort_values("Total"), y="US", x="Total",
                                orientation="h", text="Total",
                                title="💚 US 100 % Aprovadas – Prontas para Entrega",
                                color="Total", color_continuous_scale="Greens")
                fig_lv.update_traces(texttemplate="%{x} casos", textposition="outside")
                st.plotly_chart(fig_lv, use_container_width=True)
            else:
                st.info("Nenhuma User Story está 100 % aprovada ainda.")
            
        else:
            st.info("Não é possível realizar a Análise por User Story pois há apenas uma ou nenhuma US única no filtro atual.")

        # ==============================
        # RESULTADOS REPETIDOS
        # ==============================
        st.subheader("🔁 Testes com Mesmo Resultado Esperado")

        resultado_repetido = data["Resultado Esperado"].value_counts()
        duplicados = resultado_repetido[resultado_repetido > 1].index.tolist()

        if duplicados:
            rep = data[data["Resultado Esperado"].isin(duplicados)][
                ["Tipo", "TC ID", "Título do Teste", "Resultado Esperado"]
            ]
            st.dataframe(rep, height=300)
        else:
            st.info("Nenhum resultado esperado repetido encontrado.")

    else:
        st.warning("Nenhum arquivo CSV encontrado na pasta especificada.")
else:
    st.info("Informe o caminho da pasta com os arquivos CSV (ex: `dados/`).")
