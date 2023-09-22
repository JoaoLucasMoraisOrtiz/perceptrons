"""
    perceptron que dirá se a pessoa está doente ou saudável
    dados para treinar:
    entradas    esperado
    1101        1

    0010        0

    1100        0

    1011        1

    1001        0

    0011        1
"""

def perceptron(camada, entradas, bias):
    """
        efetua os calculos do neurônio
        :param camada: a camada que iremos calcular
        :type camada: array
        :param entradas: as entradas de cada conexão do neurônio
        :type entradas: array
        :param bias: o bias
        :type bias: float
        :rtype: none
    """
    
    #somatório
    somatorio = 0.0
    for neuronio in range(0, len(camada)):
        for conn in range(0, len(camada[neuronio][0])):
            somatorio += camada[neuronio][0][conn] * entradas[conn] + bias
            print(f'{somatorio} += {camada[neuronio][0][conn]} * {entradas[conn]} + {bias}')

        #função de ativação binária
        if(somatorio > 0):
            camada[neuronio][1] = 1
        else:
            camada[neuronio][1] = 0
        #exibe a saída da camada
        print(f'-=-=-=-=-=-=-=-=-=-=-=- saída camada: {camada[neuronio][1]} -=-=-=-=-=-=-=-=-=-=-=-')

def gradienteErro(saidaRede, saidaEsperada):
    """
        calcula o gradiente de erro da saída
        :param saidaRede: a saída da rede
        :type saidaRede: float
        :param saidaEsperada: a saída esperada da rede
        :type saidaEsperada: float
        :rtype: float
    """
    erro = saidaEsperada - saidaRede
    return erro

def correcaoErro(camada, aprendizado, erro, entradaNeuronio):
    """
        corrige o erro do neurônio
        :param camada: a camada que vamos corrigir o erro
        :type camada: array
        :param aprendizado: a taxa de aprendizado (fixa nesta rede)
        :type aprendizado: float
        :param erro: o erro do neurônio
        :type erro: float
        :param entradaNeuronio: as entradas que o neurônio recebeu
        :type entradaNeuronio: array
        :rtype: none
    """

    for neuronio in range(0, len(camada)):
        for conn in range(0, len(camada[neuronio][0])):
            camada[neuronio][0][conn] += + aprendizado * erro * entradaNeuronio[conn]
            print(f'erro = {camada[neuronio][0][conn]} + {aprendizado} * {erro} * {entradaNeuronio[conn]}')

def correcaoBias(bias, erro, aprendizado):
    """
        corrige o peso da bias
        :param bias: o valor atual do bias
        :type bias: float
        :param erro: o valor do erro da camada de saída
        :type erro: float
        :param aprendizado: a taxa de aprendizado da rede (fixa em nosso caso)
        :type aprendizado: float
        :rtype: float
    """
    print(f'bias = {bias} + {aprendizado} * {erro}')
    bias += aprendizado * erro
    return bias

#nossa camada de neurônios (no caso com só 1 neurônio)
neuronios1 = [
    [[0.5, 0.0, 1.5, 1.0], 0.0]
]

#o nosso bias
bias = 0

#a nossa taxa de aprendizado
aprendizado = 0.5


#frescura para ficar bonitinho
condicao = ['febre', 'enjoo', 'manchas', 'dores']

#contador de casos(d) e de eras(c)
c = 0
d = 0
while(True):

    #agrupa as entradas da camada
    paciente = []
    #resultado esperado da camada
    esperado = 0

    #se o caso é menor que o número de casos do treinamento
    if(c <= 10):
        entradas = [[[1,1,0,1],1], [[0,0,1,0],0], [[1,1,0,0],0], [[1,0,1,1],1], [[1,0,0,1],0], [[0,0,1,1],1]]
        #seleciona o d-ézimo caso de treinamento, e a d-ézima resposta correta
        paciente = entradas[d][0]
        esperado = entradas[d][1]
    else:
        #se não, pergunta sobre as condições do paciente, e recebe respostas 0 ou 1
        for count in range(0, 4):
            paciente.append(int(input(f'o paciente apresenta {condicao[count]}? [0/1] ')))
        
        #pergunta sobre como a rede deve responder (para o treinamento)
        #esperado = int(input('saida esperada [0/1]: '))
    
    #calculo da camada
    perceptron(neuronios1, paciente, bias)

    #exibe a saída esperada da camada
    print(f'-=-=-=-=-=-=-=-=-=-=-=- saída esperada {esperado} -=-=-=-=-=-=-=-=-=-=-=-')
    
    #treinamento
    erro = gradienteErro(neuronios1[0][1], esperado)
    correcaoErro(neuronios1, aprendizado, erro, paciente)
    bias = correcaoBias(bias, erro, aprendizado)

    #exibe como está o neurônio após o treinamento
    print(neuronios1[0])
    print('-=-=-=-=-=-=-=-=-=-=-=-')

    #controle de épocas e casos
    if(d == 4):
        c += 1
        d = 0
    elif(c <= 100):
        d += 1
    elif(c == 100 & d == 4):
        c += 1
        d = 0
    