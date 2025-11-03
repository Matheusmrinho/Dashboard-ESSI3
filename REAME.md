Dashboard Inteligente de Análise de Casos de Teste

Este projeto tem como objetivo analisar de forma inteligente os casos de teste funcionais, criando insights estruturais, detectando redundâncias, padrões, problemas de definição e possíveis indícios de gaps de qualidade no conjunto de testes.

Este painel foi feito em Streamlit e permite análise automática dos arquivos CSV da sprint / release, permitindo padronização e auditoria contínua do processo de QA.

🧠 O que este Dashboard entrega

Quantidade total de casos e métricas de estrutura

Verificação de presença de Pré-Condição, Steps e Resultado Esperado

Detecção de testes extremamente longos / extremamente curtos

Análise de densidade de bugs por prioridade

Identificação de casos de teste com steps muito parecidos (similaridade de texto via TF-IDF + Cosine Similarity)

Identificação de Resultados Esperados repetidos (risco de duplicidade ou redundância lógica)

📂 Estrutura do projeto
/
  dashboard.py
  requirements.txt
  dados/
     US001.csv
     US002.csv
     US003.csv
     US004.csv
     US005.csv


A pasta dados/ contém os arquivos CSV de entrada.

🚀 Como rodar

Instale dependências:

pip install -r requirements.txt


Execute o dashboard:

streamlit run dashboard.py

📁 Como adicionar novos arquivos de Sprint / Release

Gere os CSV exatamente no mesmo formato dos anteriores.

Coloque o CSV dentro da pasta dados/.

Não é necessário alterar código. O dashboard lê todos os CSV automaticamente.