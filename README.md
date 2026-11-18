# Projeto Breast Cancer - Classificacao com Machine Learning

Projeto academico de classificacao utilizando a base **Breast Cancer** da UCI Machine Learning Repository.

## Objetivo

Implementar:
- Modulo de treinamento: pre-processamento, comparacao de algoritmos, selecao do melhor modelo por acuracia media na validacao cruzada e avaliacao em teste.
- Modulo de inferencia: recebe os atributos de uma nova paciente e retorna a classe prevista no atributo `Class`.

## Sobre a base

- 286 registros.
- 9 atributos de entrada.
- Atributo alvo: `Class`.
- Classes: `no-recurrence-events` e `recurrence-events`.

**Importante:** nesta base, `Class` indica recorrencia ou nao recorrencia. Ela nao representa um diagnostico de presenca/ausencia de cancer.

Fonte: https://archive.ics.uci.edu/dataset/14/breast+cancer  
Citacao: Zwitter, M. & Soklic, M. (1988). Breast Cancer. UCI Machine Learning Repository. https://doi.org/10.24432/C51P4M.

## Estrutura

```text
projeto_breast_cancer/
|-- breast-cancer.csv
|-- treinamento.py
|-- inferencia.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

Ao executar o treinamento, serao criados:
- `modelo_breast_cancer.pkl`: pipeline do melhor modelo, utilizado na inferencia.
- `resultados_modelos.csv`: comparacao de acuracia dos algoritmos.

## Como executar

### 1. Instalar o Python

Utilize Python 3.10 ou superior.

### 2. Abrir o terminal na pasta do projeto e instalar as bibliotecas

```bash
pip install -r requirements.txt
```

### 3. Executar o treinamento

```bash
python treinamento.py
```

O programa compara Regressao Logistica, Arvore de Decisao, Random Forest, KNN, SVM e Naive Bayes. A selecao e feita pela maior acuracia media na validacao cruzada de 5 partes. Depois, o melhor pipeline e avaliado no conjunto de teste e treinado novamente com todos os dados para gerar o arquivo do modelo.

### 4. Executar a inferencia

```bash
python inferencia.py
```

Informe os dados solicitados. Os valores aceitos seguem as categorias da base UCI. Digite `?` quando nao souber um atributo; o pipeline fara a imputacao usando os dados de treinamento.

## Pre-processamento

- Valores `?` sao convertidos para valores ausentes.
- Atributos categoricos: imputacao pela categoria mais frequente e codificacao One-Hot.
- Atributo numerico `deg-malig`: imputacao pela mediana e padronizacao com StandardScaler.
- Divisao treino/teste estratificada (80%/20%).
- Validacao cruzada estratificada com 5 partes.

## Observacao

Este projeto tem finalidade exclusivamente academica. O resultado nao deve ser utilizado para diagnostico, prognostico ou decisao clinica.
