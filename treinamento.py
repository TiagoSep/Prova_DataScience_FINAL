from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA / "breast-cancer.csv"
ARQUIVO_MODELO = PASTA / "modelo_breast_cancer.pkl"
ARQUIVO_RESULTADOS = PASTA / "resultados_modelos.csv"

def carregar_dados():
    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(
            "Arquivo breast-cancer.csv não encontrado na pasta do projeto."
        )

    dados = pd.read_csv(ARQUIVO_DADOS)
    dados = dados.replace("?", np.nan)

    if "Class" not in dados.columns:
        raise ValueError("A coluna alvo 'Class' não foi encontrada.")

    X = dados.drop(columns=["Class"])
    y = dados["Class"]
    return X, y

def criar_preprocessador(X):
    colunas_categoricas = X.select_dtypes(include=["object"]).columns.tolist()
    colunas_numericas = X.select_dtypes(exclude=["object"]).columns.tolist()

    tratamento_numerico = Pipeline(steps=[
        ("imputacao", SimpleImputer(strategy="median")),
        ("normalizacao", StandardScaler())
    ])

    tratamento_categorico = Pipeline(steps=[
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("codificacao", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ("numericas", tratamento_numerico, colunas_numericas),
        ("categoricas", tratamento_categorico, colunas_categoricas)
    ])

def main():
    X, y = carregar_dados()

    print("=" * 60)
    print("TREINAMENTO - BREAST CANCER (UCI)")
    print("=" * 60)
    print(f"Quantidade de registros: {len(X)}")
    print(f"Quantidade de atributos: {X.shape[1]}")
    print("\nDistribuição das classes:")
    print(y.value_counts())

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    algoritmos = {
        "Regressao Logistica": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Arvore de Decisao": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "Naive Bayes": GaussianNB()
    }

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []
    melhor_nome = None
    melhor_acuracia_cv = -1
    melhor_pipeline = None

    print("\nComparacao dos algoritmos (acuracia media na validacao cruzada):")
    for nome, algoritmo in algoritmos.items():
        pipeline = Pipeline(steps=[
            ("preprocessamento", criar_preprocessador(X_treino)),
            ("modelo", algoritmo)
        ])

        pontuacoes = cross_val_score(
            pipeline, X_treino, y_treino,
            cv=validacao, scoring="accuracy"
        )
        media = pontuacoes.mean()
        desvio = pontuacoes.std()

        pipeline.fit(X_treino, y_treino)
        previsoes_teste = pipeline.predict(X_teste)
        acuracia_teste = accuracy_score(y_teste, previsoes_teste)

        resultados.append({
            "Modelo": nome,
            "Acuracia_CV_media": media,
            "Desvio_CV": desvio,
            "Acuracia_teste": acuracia_teste
        })

        print(
            f"{nome:22s} | CV: {media:.4f} (+/- {desvio:.4f})"
            f" | Teste: {acuracia_teste:.4f}"
        )

        if media > melhor_acuracia_cv:
            melhor_acuracia_cv = media
            melhor_nome = nome
            melhor_pipeline = pipeline

    tabela = pd.DataFrame(resultados).sort_values(
        by="Acuracia_CV_media", ascending=False
    )
    tabela.to_csv(ARQUIVO_RESULTADOS, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"Melhor modelo pela validacao cruzada: {melhor_nome}")
    print(f"Acuracia media CV: {melhor_acuracia_cv:.4f}")

    # Avaliacao final no conjunto de teste, separado do treinamento.
    previsoes = melhor_pipeline.predict(X_teste)
    print(f"Acuracia no conjunto de teste: {accuracy_score(y_teste, previsoes):.4f}")
    print("\nRelatorio de classificacao:")
    print(classification_report(y_teste, previsoes, zero_division=0))
    print("Matriz de confusao:")
    print(confusion_matrix(y_teste, previsoes))

    # Depois da avaliacao, o melhor pipeline e treinado com todos os registros
    # para ser utilizado no modulo de inferencia.
    melhor_pipeline.fit(X, y)
    joblib.dump({
        "pipeline": melhor_pipeline,
        "nome_modelo": melhor_nome,
        "atributos": X.columns.tolist(),
        "classes": sorted(y.unique().tolist())
    }, ARQUIVO_MODELO)

    print(f"\nModelo salvo em: {ARQUIVO_MODELO.name}")
    print(f"Resultados salvos em: {ARQUIVO_RESULTADOS.name}")
    print("Treinamento concluido.")

if __name__ == "__main__":
    main()
