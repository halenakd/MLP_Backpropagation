import numpy as np
import random
import csv
from scipy.optimize import minimize

def exportar_dados(filename, conjunto1, conjunto2):
    # Método para exportar dados para um arquivo csv.
    #
    # Parâmetros:
    # -- conjunto1 (): 
    # -- conjunto2 (): 

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        for i in range(len(conjunto1)):
            input_str = "[" + " ".join(map(str, conjunto1[i])) + "]"
            output_str = "[" + " ".join(map(str, conjunto2[i])) + "]"
            writer.writerow([input_str, output_str])
    
    #print("Dados exportados")

# =============================================================
# PREPARAÇÃO DOS CONJUNTOS

# CÁLCULO DAS COORDENADAS ESFÉRICAS DE CADA ÂNGULO

angles = []
coords = []

# radial coordinate r
# n-1 angular coordinates 𝜑1 , 𝜑2, … , 𝜑𝑛−1
# angles 𝜑1 , 𝜑2, … , 𝜑𝑛−2 range over [0, pi] radians (or [0, 180] degrees) and 𝜑𝑛−1 ranges over [0, 2pi] radians (or [0, 360] degrees)
# if xi are the Cartesian Coordenates, then we may compute x1, ..., xn from r, 𝜑1 , 𝜑2, … , 𝜑𝑛−1 with:
# x1 = r * cos(𝜑1),
# x2 = r * sin(𝜑1) * cos(𝜑2),
# x3 = r * sin(𝜑1) * sin(𝜑2) * cos(𝜑3),
# ...
# xn-1 = r * sin(𝜑1) * ... * sin(𝜑n-2) * cos(𝜑n-1),
# xn = r * sin(𝜑1) * ... * sin(𝜑n-2) * sin(𝜑n-1).

# sectors - 0 a 360 (longitude)
# stacks - 0 a 180 (latitude)

def cartesianCoords(r, sectors, stacks):
    coords = []
    angles = []

    for i in range(sectors):
        theta = np.radians(i * (360 / sectors))

        for j in range(stacks):
            phi = np.radians(j * (180 / stacks))

            angles.append([theta, phi])

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            coords.append([x, y, z])

    return np.array(angles), np.array(coords)

angles, coords = cartesianCoords(1.0, 72, 36)

for pattern in range(len(angles)):
    print(pattern + 1, "  input:", angles[pattern].T, "  output:", coords[pattern].T)

#exportar_dados('./esfera/coords1.csv', angles, coords)

# SEPARAÇÃO DE INPUTS E TARGETS ALEATÓRIOS

AMOSTRAS = len(angles)

random_numbers = random.sample(range(AMOSTRAS), AMOSTRAS//2)

inputs = []
targets = []

for i in random_numbers:
    inputs.append(angles[i])
    targets.append(coords[i])

#exportar_dados('./esfera/padroes.csv', inputs, targets)