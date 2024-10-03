import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from astropy.visualization import quantity_support
from scipy.optimize import curve_fit

# Definindo diretórios de entrada e saída
input_path = os.path.join(os.path.dirname(__file__), 'input')
output_path = os.path.join(os.path.dirname(__file__), 'output')

# Arquivo de dados de exemplo (pode ser alterado para outros arquivos)
source_1 = os.path.join(input_path, 'c24h12_tratado.dat')

# Função para carregar os dados de comprimento de onda e fluxo
wavel_1, flux_1 = np.loadtxt(source_1, usecols=[0, 1], unpack=True)

# Função para formatar o nome do arquivo em um rótulo no estilo LaTeX
def format_legend(filename):
    name_part = os.path.basename(filename).split('_')[0]  # Exemplo: extrair 'c24h12' do nome do arquivo
    atoms = name_part.split('h')  # Exemplo: separar 'c24' e 'h12'
    c_atoms = atoms[0][1:]  # Número de carbonos
    h_atoms = atoms[1]  # Número de hidrogênios
    return r'C$_{' + c_atoms + r'}$H$_{' + h_atoms + r'}$ - Neutral'

# Função para aplicar fatores de escala ao comprimento de onda
def apply_scale_factors(wavel, factors):
    factor1, factor2, factor3, factor4 = factors
    return np.where(wavel <= 5, wavel * factor1, 
                    np.where(wavel <= 10, wavel * factor2, 
                             np.where(wavel <= 15, wavel * factor3, 
                                      np.where(wavel > 15, wavel * factor4, wavel))))

# Aplicar os fatores de escala aos dados
wavel_1_scaled = apply_scale_factors(wavel_1, [1.04, 1.02, 1.031, 1.042])

# Configurações iniciais do gráfico
quantity_support()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel(r'Comprimento de onda [$\mu$m]', fontsize=18)
ax.set_ylabel('Intensidade [km/kmol]', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)

# Formatar e definir a legenda com base no nome do arquivo
legend_label = format_legend(source_1)
ax.plot(wavel_1_scaled, flux_1, color='#e10069', label=legend_label)

# Adicionar o cursor interativo para melhor visualização
cursor = Cursor(ax, useblit=True, horizOn=True, vertOn=True, color='gray', linewidth=1)

# Listas para armazenar os pontos marcados
marked_points_x = []
marked_points_y = []

# Função gaussiana para ajuste de curva
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Função para calcular o fluxo entre dois pontos utilizando a regra do trapézio
def calcular_fluxo(x, y):
    return np.trapezoid(y, x)

# Função de evento ao clicar no gráfico
def onclick(event):
    if event.key == 'k':  # Marcando pontos para ajuste ou cálculo de fluxo
        x = event.xdata
        y = event.ydata
        ax.scatter(x, y, color='red', marker='x', s=50)
        ax.text(x, y, f'  ({x:.2f}, {y:.2f}) km/kmol', verticalalignment='bottom', horizontalalignment='right', 
                color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        marked_points_x.append(x)
        marked_points_y.append(y)
        plt.draw()
        
    elif event.key == 'enter':  # Realizar o cálculo ao pressionar 'Enter'
        if len(marked_points_x) == 2:  # Fluxo entre dois pontos
            x1, x2 = marked_points_x
            mask = (wavel_1_scaled >= min(x1, x2)) & (wavel_1_scaled <= max(x1, x2))
            x_vals = wavel_1_scaled[mask]
            y_vals = flux_1[mask]
            fluxo = calcular_fluxo(x_vals, y_vals)
            ax.fill_between(x_vals, y_vals, color='gray', alpha=0.3, label='Fluxo do complexo')
            ax.text(np.mean([x1, x2]), np.max(y_vals) * 1.1, f'Fluxo: {fluxo:.2f} km/kmol', 
                    verticalalignment='bottom', horizontalalignment='center', color='black', fontsize=14, 
                    bbox=dict(facecolor='white', alpha=0.5))
            plt.legend()  # Atualiza a legenda
            plt.draw()
        
        elif len(marked_points_x) == 3:  # Ajuste gaussiano para três pontos
            popt, _ = curve_fit(gaussian, marked_points_x, marked_points_y, 
                                p0=[max(marked_points_y), np.mean(marked_points_x), 0.1])
            fluxo = np.sqrt(2 * np.pi) * popt[0] * np.abs(popt[2])
            ax.plot(wavel_1_scaled, gaussian(wavel_1_scaled, *popt), 'k--', label='Fluxo da emissão')
            ax.text(np.mean(marked_points_x), np.max(marked_points_y) * 1.1, 
                    f'Fluxo da emissão: {fluxo:.2f} km/kmol', verticalalignment='bottom', horizontalalignment='center', 
                    color='black', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
            plt.legend()  # Atualiza a legenda
            plt.draw()

    elif event.key == 'c':  # Limpar as marcações
        marked_points_x.clear()
        marked_points_y.clear()
        ax.clear()
        ax.plot(wavel_1_scaled, flux_1, color='#e10069', label=legend_label)
        cursor = Cursor(ax, useblit=True, horizOn=True, vertOn=True, color='gray', linewidth=1)
        plt.draw()

# Conectar o evento de clique à função onclick
fig.canvas.mpl_connect('key_press_event', onclick)

# Exibir o gráfico e salvar em um arquivo PDF
plt.legend(fontsize=14, loc='upper right')
plt.tight_layout(rect=(0, 0, 1, 1))
output_filename = os.path.join(output_path, 'plot_moleculas_interativo.pdf')
plt.savefig(output_filename)
plt.show()
