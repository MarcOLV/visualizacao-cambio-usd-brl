#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projeto de Visualização da Informação - Análise de Câmbio USD/BRL
Universidade Cruzeiro do Sul - EAD
8º Semestre

Este código implementa três visualizações diferentes baseadas no mesmo dataset:
1. Gráfico de Linhas (Unidade 2: Visualização de Informação Temporal)
2. Boxplot por Ano (Unidade 1: Estatística Descritiva)
3. Grafo de Rede (Unidade 5: Visualização de Redes e Grafos)
"""

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import networkx as nx

# Configurações gerais de visualização
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Definição de cores personalizadas
cores = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']

# Função para carregar e preparar os dados
def carregar_dados(arquivo):
    """
    Carrega e prepara o dataset para visualização
    
    Args:
        arquivo: Caminho para o arquivo CSV
        
    Returns:
        DataFrame pandas com os dados preparados
    """
    print(f"Carregando dados do arquivo: {arquivo}")
    
    # Carregamento do dataset
    df = pd.read_csv(arquivo)
    
    # Conversão da coluna de data para o formato datetime
    # Usando formato automático para evitar problemas de formatação
    df['Data'] = pd.to_datetime(df['Data'])
    
    # Verificação se as colunas Ano e Mes já existem
    if 'Ano' not in df.columns:
        df['Ano'] = df['Data'].dt.year
    if 'Mes' not in df.columns:
        df['Mes'] = df['Data'].dt.month
    
    # Adição da coluna Trimestre
    df['Trimestre'] = df['Data'].dt.quarter
    
    # Ordenação por data (mais antiga para mais recente)
    df = df.sort_values('Data')
    
    # Criação de categorias de valor para o grafo de rede
    bins = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    labels = ['1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5']
    df['Faixa_Valor'] = pd.cut(df['USD_BRL'], bins=bins, labels=labels)
    
    print(f"Dataset preparado com {len(df)} registros.")
    return df

# Função para criar o gráfico de linhas (Unidade 2: Visualização de Informação Temporal)
def criar_grafico_linhas(df, salvar_como=None):
    """
    Cria um gráfico de linhas mostrando a evolução do câmbio USD/BRL ao longo do tempo
    
    Args:
        df: DataFrame com os dados
        salvar_como: Caminho para salvar a imagem (opcional)
    """
    print("Criando gráfico de linhas (Visualização de Informação Temporal)...")
    
    # Criação da figura e eixos
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plotagem da linha principal
    ax.plot(df['Data'], df['USD_BRL'], color=cores[0], linewidth=2, label='USD/BRL Diário')
    
    # Adição de média móvel de 30 dias
    df['MM30'] = df['USD_BRL'].rolling(window=30).mean()
    ax.plot(df['Data'], df['MM30'], color=cores[5], linewidth=2, linestyle='--', 
            label='Média Móvel (30 dias)')
    
    # Adição de média móvel de 90 dias
    df['MM90'] = df['USD_BRL'].rolling(window=90).mean()
    ax.plot(df['Data'], df['MM90'], color=cores[7], linewidth=3, linestyle='-.', 
            label='Média Móvel (90 dias)')
    
    # Configuração dos eixos
    ax.set_xlabel('Data')
    ax.set_ylabel('Taxa de Câmbio (USD/BRL)')
    ax.set_title('Evolução da Taxa de Câmbio USD/BRL (2010-2019)', fontweight='bold')
    
    # Formatação do eixo x para mostrar anos
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Adição de grid e legenda
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    
    # Adição de anotações para eventos importantes
    eventos = {
        '2015-09-24': ('Rebaixamento\ndo Brasil', 4.2),
        '2016-05-12': ('Impeachment\nDilma', 3.5),
        '2018-10-28': ('Eleição\nBolsonaro', 3.7)
    }
    
    for data, (texto, y_offset) in eventos.items():
        data_evento = pd.to_datetime(data)
        idx = df[df['Data'].dt.date == data_evento.date()].index
        if len(idx) > 0:
            idx = idx[0]
            valor = df.loc[idx, 'USD_BRL']
            ax.annotate(texto, xy=(data_evento, valor), xytext=(data_evento, y_offset),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red'),
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    # Ajuste de layout
    plt.tight_layout()
    
    # Salvar a figura se um caminho for fornecido
    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo como {salvar_como}")
    
    return fig

# Função para criar o boxplot por ano (Unidade 1: Estatística Descritiva)
def criar_boxplot_anual(df, salvar_como=None):
    """
    Cria um boxplot mostrando a distribuição do câmbio USD/BRL por ano
    
    Args:
        df: DataFrame com os dados
        salvar_como: Caminho para salvar a imagem (opcional)
    """
    print("Criando boxplot anual (Visualização com gráficos de Estatística Descritiva)...")
    
    # Criação da figura e eixos
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Criação do boxplot usando seaborn
    sns.boxplot(x='Ano', y='USD_BRL', data=df, ax=ax, palette=cores[:len(df['Ano'].unique())], 
                width=0.7, linewidth=1.5, fliersize=5)
    
    # Adição de swarmplot para mostrar os pontos individuais
    # Limitando a amostra para não sobrecarregar o gráfico
    sample_df = df.groupby('Ano', group_keys=False).apply(lambda x: x.sample(min(30, len(x)))).reset_index(drop=True)
    sns.swarmplot(x='Ano', y='USD_BRL', data=sample_df, ax=ax, color='black', 
                  alpha=0.5, size=3)
    
    # Configuração dos eixos
    ax.set_xlabel('Ano')
    ax.set_ylabel('Taxa de Câmbio (USD/BRL)')
    ax.set_title('Distribuição da Taxa de Câmbio USD/BRL por Ano (2010-2019)', fontweight='bold')
    
    # Adição de estatísticas resumidas
    anos = sorted(df['Ano'].unique())
    for i, ano in enumerate(anos):
        stats = df[df['Ano'] == ano]['USD_BRL'].describe()
        ax.annotate(f"Média: {stats['mean']:.2f}\nMed: {stats['50%']:.2f}\nDP: {stats['std']:.2f}",
                    xy=(i, stats['min']), xytext=(i-0.4, stats['min']-0.15),
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # Adição de grid e ajuste de layout
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Salvar a figura se um caminho for fornecido
    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo como {salvar_como}")
    
    return fig

# Função para criar o grafo de rede (Unidade 5: Visualização de Redes e Grafos)
def criar_grafo_rede(df, salvar_como=None):
    """
    Cria um grafo de rede mostrando as relações entre anos e faixas de valor do câmbio
    
    Args:
        df: DataFrame com os dados
        salvar_como: Caminho para salvar a imagem (opcional)
    """
    print("Criando grafo de rede (Visualização de Redes e Grafos)...")
    
    # Criação da tabela de contingência para o grafo
    # Contagem de ocorrências de cada faixa de valor por ano
    network_data = pd.crosstab(df['Ano'], df['Faixa_Valor'])
    
    # Criação do grafo usando networkx
    G = nx.Graph()
    
    # Adição dos nós (anos e faixas de valor)
    anos = list(network_data.index)
    faixas = list(network_data.columns)
    
    # Adicionando nós de anos
    for ano in anos:
        G.add_node(str(ano), type='ano')
    
    # Adicionando nós de faixas de valor
    for faixa in faixas:
        G.add_node(faixa, type='faixa')
    
    # Adição das arestas (conexões entre anos e faixas de valor)
    for ano in anos:
        for faixa in faixas:
            peso = network_data.loc[ano, faixa]
            if peso > 0:
                G.add_edge(str(ano), faixa, weight=peso)
    
    # Criação da figura
    plt.figure(figsize=(14, 10))
    
    # Definição do layout do grafo
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Separação dos nós por tipo
    anos_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'ano']
    faixas_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'faixa']
    
    # Obtenção dos pesos das arestas para definir a espessura
    edge_weights = [G[u][v]['weight'] / 50 for u, v in G.edges()]
    
    # Desenho do grafo
    nx.draw_networkx_nodes(G, pos, nodelist=anos_nodes, node_color='skyblue', 
                          node_size=800, alpha=0.8, label='Anos')
    nx.draw_networkx_nodes(G, pos, nodelist=faixas_nodes, node_color='lightgreen', 
                          node_size=700, alpha=0.8, label='Faixas de Valor')
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Adição de título e legenda
    plt.title('Relações entre Anos e Faixas de Valor do Câmbio USD/BRL', fontsize=16, fontweight='bold')
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1)
    
    # Remoção dos eixos
    plt.axis('off')
    
    # Ajuste de layout
    plt.tight_layout()
    
    # Salvar a figura se um caminho for fornecido
    if salvar_como:
        plt.savefig(salvar_como, dpi=300, bbox_inches='tight')
        print(f"Grafo de rede salvo como {salvar_como}")
    
    return plt.gcf()

# Função principal
def main():
    """
    Função principal que executa todo o processo de visualização
    """
    print("Iniciando projeto de visualização da informação - Análise de Câmbio USD/BRL")
    
    # Carregamento e preparação dos dados
    df = carregar_dados('usd_brl_preparado.csv')
    
    # Criação das visualizações
    criar_grafico_linhas(df, salvar_como='grafico_linhas.png')
    criar_boxplot_anual(df, salvar_como='boxplot_anual.png')
    criar_grafo_rede(df, salvar_como='grafo_rede.png')
    
    print("Visualizações concluídas com sucesso!")

# Execução do programa
if __name__ == "__main__":
    main()
