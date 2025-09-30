import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Leitura da Tabela ---
file_path = "tabela_leandro.csv"
df = pd.read_csv(file_path)
df_filled = df.fillna(0.0)


# --- Gerar imagem da tabela original com destaque ---
def salvar_tabela_imagem(df, filename="tabela_original.png"):
    """
    Gera uma imagem da tabela do DataFrame usando matplotlib.
    - Cabeçalho destacado
    - Coluna auxiliar inteira com número da linha
    """
    # Adicionar coluna de índice como apoio (inteiro)
    df_aux = df.copy()
    df_aux.insert(0, "Linha", np.arange(1, len(df_aux) + 1, dtype=int))

    fig, ax = plt.subplots(figsize=(14, 7))  
    ax.axis('tight')
    ax.axis('off')

    tabela = ax.table(cellText=df_aux.values,
                      colLabels=df_aux.columns,
                      loc='center',
                      cellLoc='center')

    # Ajustar fonte e espaçamento
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.2, 1.2)

    # Destacar cabeçalho e coluna auxiliar
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:  # cabeçalho
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif col == 0:  # coluna "Linha"
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    return f"Tabela salva como: {filename}"


# Salvar a tabela formatada
salvar_tabela_imagem(df_filled, "tabela_formatada.png")

tabela_leandro = {int(col): df_filled[col].tolist() for col in df_filled.columns}

# --- Função para calcular upper e lower bounds e retornar q ---
def upper_lower(col_data):
    """
    col_data: lista de frequências observadas (contagens).
    Retorna (upper, lower, q) segundo Shannon (1951).
    """
    total = np.sum(col_data)
    q = np.array(col_data, dtype=float) / total
    q = np.sort(q)[::-1]  # ordenar decrescente

    # Upper bound: entropia da distribuição q
    upper = -np.sum([p * np.log2(p) for p in q if p > 0.0])

    # Lower bound: fórmula (q_i - q_{i+1}) log2(i), com q_{n+1} = 0
    q_ext = np.append(q, 0.0)
    lower = sum(i * (q_ext[i-1] - q_ext[i]) * np.log2(i)
                for i in range(1, len(q_ext)))

    return upper, lower, q

# --- Calcular para todas as colunas ---
results = {}
qs = {}
for col, values in tabela_leandro.items():
    upper, lower, q = upper_lower(values)
    results[col] = (upper, lower)
    qs[col] = q  # salvar a distribuição normalizada

# --- Impressão ---
print("Column")
print("Upper")
print("Lower")
for col in results:
    print(col, end="\t")
print()
for col in results:
    print(f"{results[col][0]:.4f}", end="\t")
print()
for col in results:
    print(f"{results[col][1]:.4f}", end="\t")
print()

# --- Função para plotar e salvar gráficos (Revisada) ---
def plot_distribution(probabilities, labels, title, filename):
    """
    Plota a distribuição de probabilidade ordenada (sticks plot) e salva em arquivo.
    
    probabilities: array/lista de probabilidades q (ordenadas).
    labels: rótulos para cada barra (ex: q_1, q_2, ...).
    title: título do gráfico.
    filename: nome do arquivo para salvar a imagem.
    """
    # 1. Configuração da Figura
    fig, ax = plt.subplots(figsize=(16, 8)) # Aumentei a largura

    n = len(probabilities)
    x_positions = np.arange(n) # Posições x padrão: 0, 1, 2, ...

    # 2. Plota os sticks
    # A largura da linha ('linewidth') foi reduzida ligeiramente para um visual mais limpo.
    ax.vlines(x_positions, 0, probabilities, colors='black', linewidth=2.5, zorder=2) 

    # 3. Valores no topo dos sticks
    for i, prob in enumerate(probabilities):
        # Ajustei a posição horizontal (+0.05) e vertical (+0.01) para melhor clareza 
        # e usei '%.3g' para mais precisão em valores pequenos.
        ax.text(x_positions[i], prob + 0.01, f'{prob:.3g}', 
                ha='center', va='bottom', fontsize=9, zorder=3,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)) # Adicionado caixa de fundo

    # 4. Limites do gráfico
    ax.set_ylim(0, max(probabilities) * 1.25) # Aumentei um pouco mais o limite superior
    # Ajustei o limite x para deixar um espaço nas bordas
    ax.set_xlim(-0.5, n - 0.5) 

    # 5. Ticks e labels do eixo x
    ax.set_xticks(x_positions)
    # Rotação de 45 graus é frequentemente melhor que 90, mas 90 é mantido se houver muitos labels.
    # Vamos usar 60 graus com alinhamento à direita para tentar um visual melhor.
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=10) 

    ax.set_xlabel('Eventos Ordenados ($q_i$)')
    ax.set_ylabel('Probabilidade')
    ax.set_title(title, fontsize=16, fontweight='bold')

    # 6. Estilo e Fundo
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(0, color='gray', linewidth=0.5) # Linha horizontal em y=0
    
    # Linhas de grade mais fracas
    plt.grid(axis='y', linestyle=':', alpha=0.5, zorder=1)
    
    # 7. Salvar
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

    return f"Gráfico salvo como: {filename}"

df_results = pd.DataFrame([
    {"N": col, "Upper": results[col][0], "Lower": results[col][1]}
    for col in results
])

# --- Função para salvar a tabela de resultados como imagem ---
def salvar_tabela_resultados(df, filename="tabela_resultados.png"):
    """
    Salva a tabela de resultados (Upper e Lower) como imagem.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    tabela = ax.table(cellText=df.values,
                      colLabels=df.columns,
                      loc='center',
                      cellLoc='center')

    # Ajustar estilo
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.2)

    # Destacar cabeçalho
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:  # cabeçalho
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    return f"Tabela de resultados salva como: {filename}"


# --- Salvar a tabela de resultados ---
salvar_tabela_resultados(df_results, "tabela_resultados.png")

# --- Gerar e salvar todos os gráficos ---
for col, q in qs.items():
    q_labels = [f'$q_{{{i}}}$' for i in range(1, len(q) + 1)]
    plot_distribution(q, q_labels, f'Distribuição Original N={col}', f'distribuicao_N{col}.png')
