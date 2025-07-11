def get_regras():
    return {
        "regra_1": "Flag se valor > 2000 e transação noturna",
        "regra_2": "Flag se device novo e valor > 1000",
    }

def aplicar_regras(row):
    # Exemplo de regra simples
    # Adapte de acordo com as features das suas bases
    if row.get('Amount', 0) > 2000:
        return 1
    return 0
