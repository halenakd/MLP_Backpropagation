# Implementação do modelo de rede Multilayer Perceptron e do algoritmo de treinamento BackPropagation
# Baseado em Hagan (2014)

import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import time

class MLP:
    def __init__(self, layers, learning_rate=0.1, epochs=10000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.biases = self.initialize_MLP()

    def initialize_MLP(self):
        # Método para inicializar as matrizes de pesos e biases da MLP.
        #
        # Retorna:
        # -- weights: pesos das conexões entre os neurônios
        # -- biases: bias de cada neurônio

        weights = []
        biases = []

        for i in range(1, len(self.layers)):        # Início em 1, pois a primeira camada é a de entrada e não está nos pesos
            weights.append(np.random.uniform(low=-0.5, high=0.5, size=(self.layers[i], self.layers[i-1]))) # intervalo [-0.5, 0.5]
            biases.append(np.random.uniform(size=(self.layers[i], 1))) # bias como valores aleatórios
        return weights, biases

    def logsigmoid(self, x):
        # Função de ativação Log-Sigmoid.
        #
        # Parâmetros:
        # -- x (): entrada "líquida da rede", a soma do bias e do produto de Wp
        #
        # Retorna:
        # -- 1 / (1 + np.exp(-x)): resultado da função de ativação Log-Sigmoid sobre o parâmetro x.

        return 1 / (1 + np.exp(-x))

    def logsigmoid_derivative(self, x):
        # Derivada da função de ativação Log-Sigmoid.
        #
        # Parâmetros:
        # -- x (): entrada "líquida da rede", a soma do bias e do produto de Wp
        #
        # Retorna:
        # -- x * (1 - x): resultado da derivada da função de ativação Log-Sigmoid sobre o parâmetro x.

        return x * (1 - x)
    
    def tanh(self, x):
        # Função de ativação Hyperbolic Tangent Sigmoid.
        #
        # Parâmetros:
        # -- x (): entrada "líquida da rede", a soma do bias e do produto de Wp
        #
        # Retorna:
        # -- np.tanh(x): resultado da função de ativação Hyperbolic Tangent Sigmoid sobre o parâmetro x.

        return np.tanh(x)

    def tanh_derivative(self, x):
        # Derivada da função de ativação Hyperbolic Tangent Sigmoid.
        #
        # Parâmetros:
        # -- x (): entrada "líquida da rede", a soma do bias e do produto de Wp
        #
        # Retorna:
        # -- 1 - np.tanh(x)**2: resultado da derivada da função de ativação Hyperbolic Tangent Sigmoid sobre o parâmetro x.

        return 1 - np.tanh(x)**2

    def purelin(self, x):
        # Função de ativação Purelin.
        #
        # Parâmetros:
        # -- x (): entrada "líquida da rede", a soma do bias e do produto de Wp
        #
        # Retorna:
        # -- x: resultado da função de ativação Purelin sobre o parâmetro x.

        return x

    def purelin_derivative(self, x):
        # Derivada da função de ativação Purelin.
        #
        # Parâmetros:
        # -- x (): entrada "líquida da rede", a soma do bias e do produto de Wp
        #
        # Retorna:
        # -- 1: resultado da derivada da função de ativação Purelin sobre o parâmetro x.

        return 1

    def forward(self, inputs):
        # Método para realizar o Forward da MLP.
        #
        # Parâmetros:
        # -- inputs (): vetor de entradas da rede
        #
        # Retorna:
        # -- a: as ativações dos neurônios da última camada da rede.

        a = inputs

        for i in range(len(self.layers)-1):
            n = self.weights[i] @ a + self.biases[i]
            a = self.logsigmoid(n) if i < len(self.layers) - 2 else self.purelin(n)
        return a

    def forward_pass(self, inputs):
        # Método para realizar o Forward Pass do algoritmo de Backpropagation da MLP.
        #
        # Parâmetros:
        # -- inputs (): vetor de entradas da rede
        #
        # Retorna:
        # -- a: todas as ativações da rede, ou seja, todas as saídas de neurônios passadas por funções de ativação.

        a = [inputs]

        for i in range(len(self.layers)-1):
            n = self.weights[i] @ a[-1] + self.biases[i]
            activation = self.logsigmoid(n) if i < len(self.layers) - 2 else self.purelin(n)
            a.append(activation)
        return a

    def backward_pass(self, inputs, targets):
        # Método para realizar o Backward Pass do algoritmo de Backpropagation da MLP.
        #
        # Parâmetros:
        # -- inputs (): vetor de vetores de entrada da rede
        # -- targets (): vetor de vetores de valores desejados da rede

        for j in range(len(inputs)):
            activations = self.forward_pass(inputs[j])
            
            sensitivities = []

            e = targets[j] - activations[-1]        # Para a última camada

            s_L = -2 * self.purelin_derivative(activations[-1]) * e
            sensitivities.append(s_L)

            self.weights[-1] = self.weights[-1] - self.learning_rate * s_L @ activations[-2].T
            self.biases[-1] = self.biases[-1] - self.learning_rate * s_L

            for i in range(len(self.weights)-2, 0-1, -1):       # Para as demais camadas
                s_i = np.diag(self.logsigmoid_derivative(activations[i+1]).flatten()) @ self.weights[i+1].T @ sensitivities[-1]
                sensitivities.append(s_i)

                self.weights[i] = self.weights[i] - self.learning_rate * s_i @ activations[i].T     # Updating weights using the approximate steep-est descent rule
                self.biases[i] = self.biases[i] - self.learning_rate * s_i      # Updating biases using the approximate steep-est descent rule

    def train(self, inputs, targets):
        # Método para realizar o treinamento da MLP através do algoritmo de Backpropagation.
        #
        # Parâmetros:
        # -- inputs (): vetor de vetores de entrada da rede
        # -- targets (): vetor de vetores de valores desejados da rede

        for epoch in range(self.epochs):
            self.backward_pass(inputs, targets)
            
            if epoch % 1000 == 0:
                total_error = 0

                for pattern in range(len(inputs)):
                    prediction = self.predict(inputs[pattern])
                    error = targets[pattern] - prediction
                    total_error += np.sum(error**2)
                
                rmse = np.sqrt(total_error / len(inputs))
                print(f"Epoch: {epoch}, RMSE: {rmse}")


    def predict(self, inputs):
        # Método para realizar a predição da MLP sobre as entradas.
        #
        # Parâmetros:
        # -- inputs (): vetor de entradas da rede
        #
        # Retorna:
        # -- self.forward(inputs): o método de Forward da rede sobre as entradas, portanto, as ativações dos neurônios da última camada da rede.

            return self.forward(inputs)
    
    def save_weights(self, weights_file, biases_file):
        # Método para salvar os pesos atuais da MLP em arquivos csv.
        #
        # Parâmetros:
        # -- weights_file (): caminho do arquivo csv de pesos
        # -- biases_file (): caminho do arquivo csv de biases

        with open(weights_file, 'w', newline='') as wf:
            writer = csv.writer(wf)
            for weight in self.weights:
                writer.writerows(weight)
                writer.writerow([])

        with open(biases_file, 'w', newline='') as bf:
            writer = csv.writer(bf)
            for bias in self.biases:
                writer.writerows(bias)
                writer.writerow([])

        print("Pesos e biases exportados")


    def load_weights(self, weights_file, biases_file):
        # Método para carregar os pesos e os biases da MLP diretamente de arquivos csv.
        #
        # Parâmetros:
        # -- weights_file (): caminho do arquivo csv de pesos
        # -- biases_file (): caminho do arquivo csv de biases

        self.weights = []
        self.biases = []
        
        with open(weights_file, 'r') as wf:
            reader = csv.reader(wf)
            weight = []
            for row in reader:
                if row:
                    weight.append([float(i) for i in row])
                else:
                    self.weights.append(np.array(weight))
                    weight = []

        with open(biases_file, 'r') as bf:
            reader = csv.reader(bf)
            bias = []
            for row in reader:
                if row:
                    bias.append([float(i) for i in row])
                else:
                    self.biases.append(np.array(bias))
                    bias = []

        print("Pesos e biases importados")


def importar_dados(filename):
    # Método para importar dados de um arquivo csv.
    #
    # Parâmetros:
    # -- filename (): vetor de entradas da rede
    #
    # Retorna:
    # -- inputs (): vetor de vetores de entrada da rede
    # -- targets (): vetor de vetores de valores desejados da rede

    inputs = []
    targets = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)

        #next(reader)  # Pula o cabeçalho
        for row in reader:
            inputs.append(np.array([[float(row[0])]]))
            targets.append(float(row[1]))

    print("Dados importados")
    
    return inputs, targets

def exportar_dados(filename, inputs, predictions, targets, errors, rmse, elapsed_time, layers, epochs, learning_rate):
    # Método para exportar dados para um arquivo csv.
    #
    # Parâmetros:
    # -- inputs ():
    # -- predictions ():
    # -- targets ():
    # -- errors ():
    # -- rmse ():
    # -- elapsed_time ():
    # -- layers ():
    # -- epochs ():
    # -- learning_rate ():

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Configurações utilizidas
        writer.writerow(['RMSE', rmse])
        writer.writerow(['Time', elapsed_time])
        writer.writerow(['Layers', layers])
        writer.writerow(['Epochs', epochs])
        writer.writerow(['Lr', learning_rate])

        writer.writerow(['Input', 'Prediction', 'Target', 'Error']) # Cabeçalho
        for i in range(len(inputs)):
            writer.writerow([inputs[i], predictions[i], targets[i], errors[i].item()])

        print("Dados exportados")


# =============================================================
# PARÂMETROS DA REDE

# neurônios de entrada, neurônios da camada 0, 1, ..., neurônios de saída
# ex.: 2, 2, 1
layers = [1, 5, 3, 1]       # Quantidade de neurônios de cada camada => define o número de camadas ocultas

learning_rate = 0.01         # Taxa de aprendizagem

epochs = 1000              # Quantidade de épocas de treinamento

# =============================================================
# PREPARAÇÃO DOS CONJUNTOS

angles, sines = importar_dados(r'seno\senos.csv')         # Importa os ângulos e senos que serão usados na predição e cálculo do erro e gráfico da rede
inputs, targets = importar_dados(r'seno\padroes.csv')     # Importa os padrões que serão utilizados no treinamento
print(inputs, targets)

# =============================================================
# CRIAÇÃO DA REDE E TREINAMENTO CRONOMETRADO

mlp = MLP(layers, learning_rate, epochs)    # Inicia a rede com os parâmetros definidos previamente

start_time = time.perf_counter()            # Inicia o cronômetro

mlp.train(inputs, targets)                  # Faz o treinamento da rede com os padrões definidos

end_time = time.perf_counter()              # Para o cronômetro 
elapsed_time = end_time - start_time        # Calcula o tempo decorrido

print(f"Tempo total de treinamento: {elapsed_time:.2f} segundos")

# =============================================================
# PASSAGEM DE TODOS OS ÂNGULOS PELA PREDIÇÃO DA REDE
# PRINT DAS INPUTS, RESULTADOS, TARGETS, E ERROS CALCULADOS

inputsplot = []
predictionsplot = []
errors = []

total_error = 0
rmse = 0

for pattern in range(len(angles)):
    prediction = mlp.predict(angles[pattern])

    error = sines[pattern] - prediction
    total_error += np.sum(error**2)

    print("Input:", angles[pattern], " Prediction:", prediction, " Target:", sines[pattern], " Error:", error)

    inputsplot.append(angles[pattern].item())
    predictionsplot.append(prediction.item())
    errors.append(error)

# CÁLCULO DO ERRO QUADRÁTICO MÉDIO

rmse = np.sqrt(total_error / len(angles))

print("RMSE:", rmse)

print(f"Tempo total de treinamento: {elapsed_time:.2f} segundos")

# EXPORT DOS RESULTADOS E DOS PESOS E BIASES FINAIS

#exportar_dados('predicoes_erros2.csv', inputsplot, predictionsplot, sines, errors, rmse, elapsed_time, layers, epochs, learning_rate)

#mlp.save_weights('pesos.csv', 'biases.csv')

# GRÁFICO

plt.plot(inputsplot, sines, label='Targets')                    # Plot dos resultados esperados
plt.plot(inputsplot, predictionsplot, label='Predictions')      # Plot dos resultados da rede
plt.legend()
plt.show()
