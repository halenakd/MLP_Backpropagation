import csv
import numpy as np
import random

# Parâmetros do pêndulo
L = 1.0  # comprimento da corda
g = 9.81  # aceleração da gravidade
m = 1.0  # massa da bolinha do pêndulo
theta0 = np.radians(45)  # ângulo inicial em radianos
omega0 = 0.0  # velocidade angular inicial
F_ext = 0.0  # força externa constante
b = 0.1  # coeficiente de resistência do ar

# Tempo
t_max = 100.0  # tempo máximo
dt = 0.01  # passo de tempo
t = np.arange(0, t_max, dt)

# Inicializar arrays
theta = np.zeros_like(t)
omega = np.zeros_like(t)

# Condições iniciais
theta[0] = theta0
omega[0] = omega0

# Método de Runge-Kutta de 4ª ordem
for i in range(1, len(t)):
    k1_theta = omega[i-1]
    k1_omega = - (g / L) * np.sin(theta[i-1]) - (b / m) * omega[i-1] + (F_ext / (m * L))
    
    k2_theta = omega[i-1] + 0.5 * k1_omega * dt
    k2_omega = - (g / L) * np.sin(theta[i-1] + 0.5 * k1_theta * dt) - (b / m) * (omega[i-1] + 0.5 * k1_omega * dt) + (F_ext / (m * L))
    
    k3_theta = omega[i-1] + 0.5 * k2_omega * dt
    k3_omega = - (g / L) * np.sin(theta[i-1] + 0.5 * k2_theta * dt) - (b / m) * (omega[i-1] + 0.5 * k2_omega * dt) + (F_ext / (m * L))
    
    k4_theta = omega[i-1] + k3_omega * dt
    k4_omega = - (g / L) * np.sin(theta[i-1] + k3_theta * dt) - (b / m) * (omega[i-1] + k3_omega * dt) + (F_ext / (m * L))
    
    theta[i] = theta[i-1] + (dt / 6.0) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    omega[i] = omega[i-1] + (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

# Coordenadas x e y da bolinha do pêndulo
x = L * np.sin(theta)
y = -L * np.cos(theta)

# Exportar dados
def exportar_dados(filename, conjunto1, conjunto2):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(conjunto1)):
            writer.writerow([conjunto1[i], conjunto2[i]])

# Criar arrays de inputs e targets
angulos_velocidadesAngulares = np.column_stack((theta, omega))
coords = np.column_stack((x, y))

exportar_dados('./pendulo/angulos_velocidadesAngulares.csv', angulos_velocidadesAngulares, coords)

# SEPARAÇÃO DE INPUTS E TARGETS ALEATÓRIOS

AMOSTRAS = len(angulos_velocidadesAngulares)
print(AMOSTRAS)

random_numbers = random.sample(range(AMOSTRAS), AMOSTRAS//10)
print(AMOSTRAS//10)

inputs = []
targets = []

for i in random_numbers:
    inputs.append(angulos_velocidadesAngulares[i])
    targets.append(coords[i])

exportar_dados('./pendulo/padroes.csv', inputs, targets)