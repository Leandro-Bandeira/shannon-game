import numpy as np

# Dados da Tabela I (cada coluna corresponde a um N)
# Cada coluna deve somar 100 (percentual). As linhas representam tentativas (i=1..27).
tabela = {
    1: [18.2, 10.7, 8.6, 6.7, 6.5, 5.8, 5.6, 5.2, 5.0, 4.3, 3.1, 2.8, 2.4, 2.3, 2.1, 2.0, 1.6, 1.6, 1.6, 1.3, 1.2, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1],
    2: [29.2, 14.8, 10.0, 8.6, 7.1, 5.5, 4.5, 3.6, 3.0, 2.6, 2.2, 1.9, 1.5, 1.2, 1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.0, 0, 0, 0],
    3: [37.6, 17.6, 11.0, 7.3, 5.0, 3.8, 3.1, 2.5, 2.0, 1.7, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    4: [45.1, 18.0, 11.2, 7.1, 4.7, 3.4, 2.4, 1.8, 1.4, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    5: [50.8, 18.6, 11.3, 6.7, 4.1, 2.7, 1.7, 1.2, 0.9, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    6: [57.9, 19.0, 10.9, 6.0, 3.4, 2.0, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

def upper_lower(col_data):
    q = np.array(col_data, dtype=float) / 100.0
    # Upper bound: entropia da distribuição q
    upper = -np.sum([p * np.log2(p) for p in q if p > 0.0])
    # Lower bound (fórmula corrigida de Shannon)
    q_ext = np.append(q, 0.0)
    lower = sum(i* (q_ext[i-1] - q_ext[i]) * np.log2(i) for i in range(1, len(q_ext)))
    return upper, lower

# Calcular para todas as colunas
results = {}
for col, values in tabela.items():
    results[col] = upper_lower(values)

# Impressão no formato desejado
print("Column")
print("Upper")
print("Lower")

for col in results:
    print(col, end="\t")
print()
for col in results:
    print(f"{results[col][0]:.2f}", end="\t")
print()
for col in results:
    print(f"{results[col][1]:.2f}", end="\t")
print()