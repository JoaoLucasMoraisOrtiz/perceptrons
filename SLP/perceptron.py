"""
    Rede Perceptron simples. Treine a rede com os casos de teste, e ela vai conseguir sempre acerta-los!
    Casos (x1, x2, x3 são as entradas de cada caso, e d a resposta esperada):
    x1 1 1 1 1
    x2 0 0 1 1
    x3 0 1 0 1
    d  0 0 0 1
"""

def perceptron(neuronios, entradas, bias):
    """
        efetua os calculos do neurônio
        :param neuronios: a camada que iremos calcular
        :type neuronios: array
        :param entradas: as entradas de cada conexão do neurônio
        :type entradas: array
        :param bias: o bias
        :type bias: float
        :rtype: none
    """
    somatorio  = float()

    #somatório
    for neuronio in range(0, len(neuronios)):
        for conn in range(0, len(neuronios[neuronio]) + 1):
            somatorio += neuronios[neuronio][0][conn] * entradas[neuronio][conn] + bias
            print(f'somatorio = {neuronios[neuronio][0][conn]} * {entradas[neuronio][conn]} = {somatorio}')
        if(somatorio > 0):
            neuronios[neuronio][1] = 1
        else:
            neuronios[neuronio][1] = 0

def saida(neuronios):
    """
		retorna a resposta de uma camada de neurônios
		:param neuronios: a camada que queremos pegar o resultado
		:type neuronios: array
		:rtype: float
	"""
    s = float()
    for neuronio in neuronios:
        s += neuronio[1]
    return s

def coeficientErro(neuronio, esperado):
    """
        calcula o gradiente de erro da saída
        :param neuronio: o neurônio da camada de saída
        :type neuronio: array
        :param saidaEsperada: a saída esperada da rede
        :type saidaEsperada: float
        :rtype: float
    """
    print(f'erro = {esperado} - {neuronio[1]}')
    error = esperado - neuronio[1]
    return error

def bp(neuronios, aprendizado, erro, entrada):
    """
        corrige o erro do neurônio
        :param neuronios: a camada que vamos corrigir o erro
        :type neuronios: array
        :param aprendizado: a taxa de aprendizado (fixa nesta rede)
        :type aprendizado: float
        :param erro: o erro do neurônio
        :type erro: float
        :param entrada: as entradas que o neurônio recebeu
        :type entrada: array
        :rtype: none
    """
    for neuronio in range(0, len(neuronios)):
        for conn in range(0, len(neuronios[neuronio][0])):
            print(f'{neuronios[neuronio][0][conn]} + {aprendizado} * {erro} * {entrada[neuronio][conn]} = {aprendizado*erro*entrada[neuronio][conn]}')
            neuronios[neuronio][0][conn] = neuronios[neuronio][0][conn] + aprendizado*erro*entrada[neuronio][conn]

def biasErros(bias, aprendizado, erro):
    """
        corrige o peso da bias
        :param bias: o valor atual do bias
        :type bias: float
        :param aprendizado: a taxa de aprendizado da rede (fixa em nosso caso)
        :type aprendizado: float
        :param erro: o valor do erro da camada de saída
        :type erro: float
        :rtype: float
    """
    bias += aprendizado * erro
    return bias

#declaração de variáveis gerais
bias = 0
neuronios1 = [
    [[0, 0, 0], 0]
]

while(True):

    #declaração de variáveis mutáveis
    entrada = []
    esperados = int()

    #limpa o último calculo dos neurônios
    for neuronio in neuronios1:
        neuronio[1] = 0
    
    c = 1
    #loop das entradas
    while(c <= 1):
        d = 1
        caso = list()
        while(d <= 3):
            caso.append(int(input(f'Digite o {d}º dado de entrada: ')))
            d += 1
        entrada.append(caso)
        d = 1
        esperados = (int(input(f'Digite a saída esperada do caso {c}: ')))
        c += 1
    del c
    del d

    #garante o fim da execução do programa caso digite 999
    if(entrada == [[9,9,9]]):
        for neuronio in neuronios1:
            print(neuronio)
        print('agora deu certo ???')
        exit()

    #calculo da camada
    perceptron(neuronios1, entrada, bias)

    #saída da camada
    resultado = saida(neuronios1)
    if(resultado <= 0):
        resultado = 0
    else:
        resultado = 1

    #exibe as saídas da camada
    print(f'-=-=-=-=-=-=-=-=-=- saída da rede: {resultado} -=-=-=-=-=-=-=-=-=-')
    print(f'-=-=-=-=-=-=-=-=-=- saída esperada: {esperados} -=-=-=-=-=-=-=-=-=-')
    print(neuronios1[0])

    #TREINAMENTO

    aprendizado = 0.5
    erro = 0.0

    #calcula o coeficiente de erro
    erro = coeficientErro(neuronios1[0], esperados)

    #faz a propagação do erro
    bp(neuronios1, aprendizado, erro, entrada)

    #corrige o bias
    biasErros(bias, aprendizado, erro)