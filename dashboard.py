import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Dashboard Inteligente de Testes", layout="wide")

st.title("🧩 Dashboard de Qualidade de Testes")
st.markdown("""
Este painel foca nas métricas de **execução e qualidade**, como Taxa de Sucesso e Resolução de Bugs, agrupados por **Arquivo/US**.
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
                # Tenta ler com diferentes separadores e encodings
                df = pd.read_csv(os.path.join(data_dir, file), encoding="utf-8", sep=None, engine="python")
            except UnicodeDecodeError:
                df = pd.read_csv(os.path.join(data_dir, file), encoding="latin1", sep=None, engine="python")

            # Mapeamento de colunas (Mantemos 'Story Link' aqui apenas para alinhar a leitura, mas vamos removê-la logo abaixo)
            df.columns = [
                "Story Link", "TC ID", "Título do Teste", "Pré-condição",
                "Passos", "Resultado Esperado", "Resultado Execução",
                "Bug", "Prioridade"
            ]
            
            # 🗑️ REMOÇÃO DO STORY LINK
            # Removemos a coluna imediatamente para não usar dados sujos (MUN-xxx, etc)
            df.drop(columns=["Story Link"], inplace=True)

            df["Arquivo"] = file

            # ==============================
            # DETECTA TIPO DE TESTE
            # ==============================
            # Verifica se o nome do arquivo termina com US seguido por 3 dígitos (Unitário)
            if file.lower().split(".csv")[0].endswith(tuple([f"us{str(i).zfill(3)}" for i in range(1,999)])):
                df["Tipo"] = "Unitário"
            else:
                df["Tipo"] = "Regressão"

            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        # Aplica strip em todas as strings para limpar espaços
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # ==============================
        # TRATAMENTO E LIMPEZA DE DADOS
        # ==============================

        # Limpa o nome dos arquivos para exibição na sidebar e para usar como ID da US
        prefixo_comum = "[entrenos] Execução de testes 2025.1 -"
        data["Arquivo Limpo"] = data["Arquivo"].str.replace(prefixo_comum, "", regex=False).str.strip()
        
        # 🆕 CRIAÇÃO DA COLUNA 'US' BASEADA NO ARQUIVO
        # Como removemos o Story Link, usamos o nome do arquivo (sem extensão) como identificador da US
        data["US"] = data["Arquivo Limpo"].str.replace(".csv", "", regex=False)

        # ==============================
        # FILTRO POR TIPO
        # ==============================
        st.sidebar.subheader("Tipos de Teste")
        tipos = st.sidebar.multiselect(
            "Filtrar por tipo",
            ["Unitário", "Regressão"],
            default=["Unitário", "Regressão"]
        )
        data = data[data["Tipo"].isin(tipos)]

        # ==============================
        # FILTRO POR ARQUIVO CSV
        # ==============================
        st.sidebar.subheader("Filtro por User Story (Arquivo)")

        arquivos_limpos_list = sorted(data["Arquivo Limpo"].dropna().unique().tolist())
        arquivos_selecionados_limpos = st.sidebar.multiselect(
            "Selecione o Arquivo CSV", 
            arquivos_limpos_list, 
            default=arquivos_limpos_list
        )

        if len(arquivos_selecionados_limpos) > 0:
            data = data[data["Arquivo Limpo"].isin(arquivos_selecionados_limpos)]
        else:
            data = data.head(0)
            st.warning("Selecione pelo menos um arquivo para visualizar os dados.")
        
        # Só prossegue se o DataFrame não estiver vazio após os filtros
        if data.empty:
            st.info("Nenhum dado restante após a aplicação dos filtros.")
        else:
            
            # ==============================
            # MÉTRICAS DERIVADAS
            # ==============================
            
            # Novo: Bug Reportado (Campo 'Bug' está preenchido)
            data["Tem_Bug_Reportado"] = data["Bug"].notna() & (data["Bug"].astype(str).str.strip() != "")
            # A coluna 'Tem Bug' agora é um alias
            data["Tem Bug"] = data["Tem_Bug_Reportado"] 
            
            # Coluna para bugs verificados como resolvidos (TESTE PASSOU E TINHA BUG REPORTADO)
            data["Bug_Verificado_Resolvido"] = (
                data["Resultado Execução"].str.contains("PASSED|OK|SUCESS", case=False, na=False) &
                data["Tem_Bug_Reportado"]
            )
            
            # Bugs Reportados (FAILED que tem link) - Usada na taxa
            data["Bugs_Reportados"] = (
                data["Resultado Execução"].str.contains("FAILED|ERRO|FALHA", case=False, na=False) &
                data["Tem_Bug_Reportado"]
            )
            
            # Necessário para o card "Sem Resultado Esperado"
            data["Tem Resultado Esperado"] = data["Resultado Esperado"].notna()

            # ==============================
            # MÉTRICAS GERAIS POR TIPO
            # ==============================
            st.subheader("📊 Métricas por Tipo de Teste")

            for tipo in ["Unitário", "Regressão"]: 
                subset = data[data["Tipo"] == tipo]
                if len(subset) == 0:
                    continue

                st.markdown(f"## 🔹 {tipo}")
                
                # Cálculo de métricas específicas para este subconjunto (tipo)
                total = len(subset)
                passaram = subset["Resultado Execução"].str.contains("PASSED|OK|SUCESS", case=False, na=False).sum()
                falharam = subset["Resultado Execução"].str.contains("FAILED|ERRO", case=False, na=False).sum()
                bugs = subset["Tem Bug"].sum()
                resolvidos_subset = subset["Bug_Verificado_Resolvido"].sum()

                if tipo == 'Unitário':
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Total", total)
                    col2.metric("Passaram", passaram)
                    col3.metric("Falharam", falharam)
                    col4.metric("Bugs", bugs)
                    col5.metric("Sem Resultado Esperado", len(subset) - subset["Tem Resultado Esperado"].sum())
                    col6.metric("Bugs Verificados Resolvidos", resolvidos_subset)
                else:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Total", total)
                    col2.metric("Passaram", passaram)
                    col3.metric("Falharam", falharam)
                    col4.metric("Bugs", bugs)
                    col5.metric("Sem Resultado Esperado", len(subset) - subset["Tem Resultado Esperado"].sum())

                st.divider()


            # ==============================
            # BUGS POR PRIORIDADE
            # ==============================
            st.subheader("🐞 Densidade de Bugs por Prioridade")

            # 1. Filtro Robusto
            data_filtrada = data.dropna(subset=["Prioridade"])
            data_filtrada = data_filtrada[
                data_filtrada["Prioridade"].astype(str).str.strip().str.upper() != "PRIORIDADE"
            ]

            if not data_filtrada.empty:
                # 2. Agrupa e calcula a média
                bugs_por_prioridade = data_filtrada.groupby("Prioridade")["Tem Bug"].mean().reset_index()
                bugs_por_prioridade["Tem Bug"] *= 100
                
                # 3. Verifica e Plota
                if len(bugs_por_prioridade) > 0:
                    fig3 = px.bar(
                        bugs_por_prioridade, 
                        x="Prioridade", 
                        y="Tem Bug",
                        title="Percentual de Testes com Bug por Prioridade", 
                        text_auto=".1f"
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Não há dados de Prioridade válidos para esta análise.")
            else:
                st.info("Não há testes com prioridade definida para calcular a densidade de bugs.")


            if data["US"].nunique() > 0: # Ajustado para > 0 pois agora US sempre existe (é o nome do arquivo)
        
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
                                 title="Taxa de Sucesso por User Story (Baseado no Arquivo)",
                                 color="Taxa de Sucesso (%)", color_continuous_scale="RdYlGn")
                fig_evo.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_evo, use_container_width=True)

                # ---------------------------------------------------------

                # RESULTADOS POR US
                st.subheader("📉 Distribuição de Resultados por US")
                
                data["Status_Agrupado"] = data["Resultado Execução"].apply(lambda x: 
                    "PASSED" if str(x).upper() in ["PASSED", "OK", "SUCESS"] else
                    "FAILED" if str(x).upper() in ["FAILED", "ERRO", "FALHA"] else
                    "NÃO EXECUTADO")
                    
                res_us = data.groupby(["US", "Status_Agrupado"]).size().unstack(fill_value=0)
                
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
                taxa_df = (
                data.groupby("US")
                .agg(
                    Bugs_Reportados=("Bugs_Reportados", "sum"), 
                    Bugs_Verificados=("Bug_Verificado_Resolvido", "sum") 
                )
                .assign(
                    Taxa_Resolucao=lambda df: (
                        (df["Bugs_Verificados"] / df["Bugs_Reportados"].replace(0, 1))
                        .clip(upper=1.0) * 100
                    ).round(1)
                )
                .fillna(0)
                .reset_index()
                )

                st.subheader("📈 Taxa de Resolução de Bugs por US")
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
                    st.success("Nenhum teste crítico de Prioridade ALTA falhou.")

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

                # 1. Cria uma cópia para trabalhar os grupos
                df_entrega = data.copy()

                # 2. Aplica a regra: Se não começar com "US", vira "Regressão"
                def agrupar_regressao(nome_us):
                    nome_us = str(nome_us).strip().upper()
                    # Se começar com US (ex: US001, US-001), mantém o nome. Senão, agrupa.
                    if nome_us.startswith("US"):
                        return nome_us
                    return "Regressão"

                df_entrega["Grupo_Entrega"] = df_entrega["US"].apply(agrupar_regressao)

                # 3. Calcula os dados agrupados
                sumario_entrega = df_entrega.groupby("Grupo_Entrega").agg(
                    Total=("TC ID", "count"),
                    Passaram=("Status_Agrupado", lambda x: (x == "PASSED").sum())
                ).reset_index()

                # Calcula a taxa de sucesso
                sumario_entrega["Taxa de Sucesso (%)"] = (sumario_entrega["Passaram"] / sumario_entrega["Total"].replace(0, 1) * 100).round(1)

                # 4. Plota TUDO (não filtramos mais só os 100%, para você ver as USs com falha também)
                if not sumario_entrega.empty:
                    # Ordena: Regressão no topo ou base, e o resto pela taxa de sucesso
                    sumario_entrega = sumario_entrega.sort_values(by=["Taxa de Sucesso (%)", "Total"], ascending=True)
                    
                    fig_lv = px.bar(sumario_entrega, y="Grupo_Entrega", x="Taxa de Sucesso (%)",
                                    orientation="h", 
                                    text="Taxa de Sucesso (%)",
                                    title="🚦 Status de Entrega: Regressão vs User Stories",
                                    labels={"Grupo_Entrega": "Pacote de Entrega", "Taxa de Sucesso (%)": "% Aprovado"},
                                    color="Taxa de Sucesso (%)", 
                                    color_continuous_scale="RdYlGn", # Vermelho -> Amarelo -> Verde
                                    range_color=[0, 100]) # Garante que 0 é vermelho e 100 é verde
                    
                    # Adiciona informação de quantos testes existem em cada barra
                    fig_lv.update_traces(
                        texttemplate="%{x}% (%{customdata[0]} testes)", 
                        customdata=sumario_entrega[["Total"]],
                        textposition="inside"
                    )
                    
                    st.plotly_chart(fig_lv, use_container_width=True)
                else:
                    st.info("Não há dados suficientes para gerar o status de entrega.")

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
                st.dataframe(rep, use_container_width=True)
            else:
                st.info("Nenhum resultado esperado repetido encontrado.")

    else:
        st.warning("Nenhum arquivo CSV encontrado na pasta especificada.")
else:
    st.info("Informe o caminho da pasta com os arquivos CSV (ex: `dados/`).")