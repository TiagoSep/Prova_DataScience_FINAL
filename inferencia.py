from pathlib import Path
import joblib
import pandas as pd

PASTA = Path(__file__).resolve().parent
ARQUIVO_MODELO = PASTA / "modelo_breast_cancer.pkl"

CAMPOS = {
    "age": ("Faixa etaria (ex.: 40-49)", ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]),
    "menopause": ("Menopausa", ["lt40", "ge40", "premeno"]),
    "tumor-size": ("Tamanho do tumor", ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"]),
    "inv-nodes": ("Numero de linfonodos invadidos", ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"]),
    "node-caps": ("Capsula dos linfonodos", ["yes", "no"]),
    "deg-malig": ("Grau de malignidade (1, 2 ou 3)", ["1", "2", "3"]),
    "breast": ("Mama", ["left", "right"]),
    "breast-quad": ("Quadrante da mama", ["left-up", "left-low", "right-up", "right-low", "central"]),
    "irradiat": ("Radioterapia", ["yes", "no"])
}

def obter_valor(campo, descricao, opcoes):
    opcoes_texto = ", ".join(opcoes)
    while True:
        valor = input(f"{descricao} [{opcoes_texto}] (ou ? se desconhecido): ").strip()
        if valor == "?":
            return None
        if valor in opcoes:
            return int(valor) if campo == "deg-malig" else valor
        print("Valor invalido. Digite uma das opcoes apresentadas ou ?.")

def main():
    if not ARQUIVO_MODELO.exists():
        print("Modelo nao encontrado. Execute primeiro: python treinamento.py")
        return

    pacote = joblib.load(ARQUIVO_MODELO)
    modelo = pacote["pipeline"]

    print("=" * 60)
    print("INFERENCIA - NOVA PACIENTE")
    print("=" * 60)
    print(f"Modelo utilizado: {pacote['nome_modelo']}")
    print("Preencha os dados conforme as categorias da base UCI.\n")

    paciente = {}
    for campo, (descricao, opcoes) in CAMPOS.items():
        paciente[campo] = obter_valor(campo, descricao, opcoes)

    entrada = pd.DataFrame([paciente], columns=pacote["atributos"])
    classe = modelo.predict(entrada)[0]

    print("\n" + "=" * 60)
    print("RESULTADO DA CLASSIFICACAO")
    print("=" * 60)
    print(f"Classe prevista: {classe}")

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(entrada)[0]
        print("\nProbabilidades estimadas pelo modelo:")
        for nome_classe, probabilidade in zip(modelo.classes_, probabilidades):
            print(f"- {nome_classe}: {probabilidade:.2%}")

    print("\nObservacao: esta classificacao estima recorrencia conforme")
    print("as classes historicas desta base. Nao e diagnostico medico.")

if __name__ == "__main__":
    main()
