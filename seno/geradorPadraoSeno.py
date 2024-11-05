import numpy as np
import random
import csv

def exportar_dados(filename, conjunto1, conjunto2):
    # Método para exportar dados para um arquivo csv.
    #
    # Parâmetros:
    # -- conjunto1 (): 
    # -- conjunto2 (): 

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        #writer.writerow(['Input', 'Target']) # Cabeçalho
        for i in range(len(conjunto1)):
            writer.writerow([conjunto1[i].item(), conjunto2[i].item()])
    
    #print("Dados exportados")

# =============================================================
# PREPARAÇÃO DOS CONJUNTOS

# CÁLCULO DO SENO DE CADA ÂNGULO

angles = []
sines = []

for i in range(0, 361):
    seno = np.sin(np.radians(i))
    angulo = np.array([[np.radians(i)]])
    angles.append(angulo)
    sines.append(seno)

for pattern in range(len(angles)):
    print(pattern + 1, "  input:", angles[pattern].T, "  output:", sines[pattern].T)

exportar_dados('./seno/senos.csv', angles, sines)

# SEPARAÇÃO DE INPUTS E TARGETS ALEATÓRIOS

AMOSTRAS = 180

random_numbers = random.sample(range(361), AMOSTRAS)

inputs = []
targets = []

for i in random_numbers:
    inputs.append(angles[i])
    targets.append(sines[i])

exportar_dados('./seno/padroes.csv', inputs, targets)