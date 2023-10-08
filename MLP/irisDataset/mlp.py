"""
    MLP para fazer a separação do datase Iris
"""
import math
from random import shuffle, randint

def handleIris(patch):
    """
        retorna uma string contendo as entradas e respostas esperadas do Iris dataset.
        Já deixa padronizada a saída para Iris-setosa = [1, 0, 0], Iris-versicolor = [0, 1, 0] e Iris-virginica = [0, 0, 1].
        :param patch: caminho até o arquivo iris.data.
        :type patch: string.
        :rtypr: array.
    """
    doc = open(patch, 'r')
    data = []
    for line in doc:
        data.append([[float(line.split(',')[i])] for i in range(0, len(line.split(',')) -1)])
        data[-1].append(line.split(',')[-1])

        if data[-1][-1] == 'Iris-setosa\n':
            data[-1][-1] = [1, 0, 0]
            #data.pop()
        elif data[-1][-1] == 'Iris-versicolor\n':
            data[-1][-1] = [0, 1, 0]
            #data.pop()
        else:
            #data.pop()
            data [-1][-1] = [0, 0, 1]

    return data

def funcaoAtivacao1(x):
    """
        função de ativação da camada oculta
        :param x: o valor de x para a função de ativação
        :type x: float
    """
    #x = round(x, 30)
    """ if(abs(x) > 489):
        x = x*490/abs(x) """
    r = 1/(1+(math.e)**(-x))
    return r

def derivadaFuncAtivacao1(y):
    """
        função que retorna a derivada da função de ativação com y = f(x) 
        :param x: o ponto onde x deve ser derivado
        :type x: float
    """
    derivada = y * (1.0 - y)
    return derivada

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
    arrSomatorios = []
    for neuronio in range(0, len(camada)):
        #somatório
        somatorio = 0.0

        for conn in range(0, len(camada[neuronio][0])):
            try:
                #print(f'{somatorio} += {camada[neuronio][0][conn]} * {entradas[neuronio]} + {bias}')
                somatorio += camada[neuronio][0][conn] * entradas[neuronio][conn] + bias
            except:
                try:
                    somatorio += camada[neuronio][0][conn] * entradas[conn] + bias
                except:
                    somatorio += camada[neuronio][0][conn] * entradas[conn][0] + bias
        """ if(abs(somatorio) > 50):
            somatorio = (somatorio*50)/abs(somatorio) """

        #função de ativação
        r = funcaoAtivacao1(somatorio)
        camada[neuronio][1] = r
        #exibe a saída do neurônio 'n' da camada
        #print(f'-=-=-=-=-=-=-=-=-=-=-=- saída do {neuronio}º neuronio da camada: {camada[neuronio][1]} -=-=-=-=-=-=-=-=-=-=-=-')
        
        #guarda o valor do somatório do n-ézimo neurônio
        arrSomatorios.append(somatorio)
    return arrSomatorios

def respostaCamada(camada):
	"""
		retorna a resposta de uma camada de neurônios
		:param camada: a camada que queremos pegar o resultado
		:type camada: array
		:rtype: array
	"""
	resp = []
	for neuronio in range(0, len(camada)):
		resp.append(camada[neuronio][1])
	return resp

def gradienteErroSaida(saidaRede, saidaEsperada):
    """
        calcula o gradiente de erro da saída, segundo proposto pela professora Ariane Machado Lima (USP):
        https://edisciplinas.usp.br/pluginfile.php/4457290/mod_resource/content/2/SIN5007-Tema08-RedesNeurais.pdf
        :param saidaRede: a saída da rede
        :type saidaRede: float
        :param saidaEsperada: a saída esperada da rede
        :type saidaEsperada: float
        :param somatorioNeuronio: os somatórios dos neurônios antes de passarem pela função de ativação
        :type somatorioNeuronio: array.
        :rtype: array
    """
    erro = []
    for item in range(0, len(saidaEsperada)):

        erro.append((saidaEsperada[item] - saidaRede[item]) * derivadaFuncAtivacao1(saidaRede[item]))
    return erro

def gradienteErroOculto(camadaSaida, erroSaida, posNeuronioOculto, somatorioOculto):
    """
        calcula o gradiente de erro de um neurônio j da camada oculta.
        faz a soma ponderada de todas as conexões que este neurônio faz com a camada de saída e retorna este resultado multiplicado pela f'(x)
        :param camadaSaida: a camada de saida da rede, com seus neurônios.
        :type saidaRede: array.
        :param erroSaida: os erros calculados da camada de saida da rede.
        :type saidaEsperada: array.
        :param numeroNeuronioOculto: a posição do neurônio oculto na camada oculta.
        :type numeroNeuronioOculto: int.
        :param somatorioNeuronio: o somatório da camada oculta no neuronio que estamos pegando o erro.
        :type somatorioNeuronio: float.
        :rtype: float
    """
    somatorio = 0.0
    #para cada neurônio na camada de saída que faz conexão com o neurônio da camada oculta
    for neuronioSaida in range(0, len(camadaSaida)):
        #somatório (que é negativo), soma o erro da saida do neurônioSaida multiplicado pelo peso da sua conexão com o neurônio oculto
        somatorio += erroSaida[neuronioSaida] * camadaSaida[neuronioSaida][0][posNeuronioOculto]
    
    #retorna o somatório vezes a derivada da função de ativação em x = somatório do neurônio oculto
    erro = somatorio * derivadaFuncAtivacao1(funcaoAtivacao1(somatorioOculto[posNeuronioOculto]))
    return erro

def correcaoErro(camada, aprendizado, erro, entradaNeuronio):
    """
        corrige o erro dos neurônios
        :param camada: a camada que vamos corrigir o erro
        :type camada: array
        :param aprendizado: a taxa de aprendizado (fixa nesta rede)
        :type aprendizado: float
        :param erro: os erros dos neurônios da camada
        :type erro: array
        :param entradaNeuronio: as entradas que o neurônio recebeu
        :type entradaNeuronio: array
        :rtype: none
    """

    for neuronio in range(0, len(camada)):
        for conn in range(0, len(camada[neuronio][0])):
            #print(f'erro = {camada[neuronio][0][conn]} + {aprendizado} * {erro[neuronio]} * {entradaNeuronio[conn]}')
            try:
                delta = + aprendizado * erro[neuronio] * entradaNeuronio[conn]
                camada[neuronio][0][conn] += delta
            except:
                delta = + aprendizado * erro[neuronio] * entradaNeuronio[conn][0]
                camada[neuronio][0][conn] += delta

def correcaoBias(bias, respCamadaSaida, esperado):
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
    #print(f'bias = {bias} + {aprendizado} * {erro}')
    sumError = 0.0
    
    for item in range(0, len(respCamadaSaida)):
        sumError += (esperado[item] - respCamadaSaida[item]) ** 2

    bias += aprendizado * (1/2 * sumError) * derivadaFuncAtivacao1(funcaoAtivacao1(sumError))
    #bias += aprendizado * sumError
    return bias

def resultados(respCamadaSaida, esperado):
    print(f'{respCamadaSaida} ?= {esperado}')

    if respCamadaSaida[0] > respCamadaSaida[1] and respCamadaSaida[0] > respCamadaSaida[2]:
        a = [1, 0, 0]
    elif respCamadaSaida[1] > respCamadaSaida[0] and respCamadaSaida[1] > respCamadaSaida[2]:
        a = [0, 1, 0]
    else:
        a = [0, 0, 1]
    
    #exibe a saída esperada da camada
    if(a == esperado):
        print(f'ACERTOU')
        return 1
    else:
        print(f'ERROU')
        return 0
#[, 1.928749847963923e-22]
#camada de entrada
neuroniosEntrada = [
    [[-189540724451.10132], 0.0],
    [[0.0], 0.0],
    [[-189364613433.16177], 0.0],
    [[0.0], 0.0]
]

neuronioOculto = [
    [[-0.1332716458, 0.1286843714, 0.4578311077, -0.0828125646], 0.0],
    [[-0.0965339492, -0.0888721601, 0.0674008359, 0.273107275], 0.0],
    [[1.1446857578, 0.2068841463, 0.5627325617, 0.6588644044], 0.0],
    [[0.1246586779, 0.89198765, 0.4765024435, 0.6275846291], 0.0],
    [[834.3877172944, 3463.5125318272, 831.5751221024, 3463.716549207], 0],
    [[425.2914915577, 7.2534970734, 410.2249621772, 6.8729185236], 0.0],
    [[-0.0773061666, -0.167516406, 0.0543271937, 0.3031860352], 0.0],
    [[0.0357049222, -0.0334898736, -0.114429559, 0.2997972207], 0.0],
    [[0.1662754607, 0.0341925282, -0.0512390273, 0.0638606023], 0.0],
    [[1.1388856093, 0.2637333896, 0.2828034977, -0.1763537536], 0.0],
    [[518.6173344035, 3365.1675824212, 538.3343852509, 3365.3301038666], 0.0],
    [[0.6755318834, 0.1819658377, 0.6509834012, 0.5053309105], 0.0],
    [[0.2342829686, -0.1037659124, 0.6758371005, 0.4908106857], 0.0],
    [[0.1048324952, -0.0508065369, 0.243878566, 0.1289279044], 0.0],
    [[0.1798745306, 0.0322275749, 0.6106355075, 0.0397609247], 0.0],
    [[0.6101088358, 0.1981202964, 0.0932446752, -0.0776486692], 0.0],
    [[0.0056529164, 0.0999808228, 0.1380425319, -0.1075664863], 0.0],
    [[0.4956879021, 0.2861726139, -0.0003620125, -0.1427854548], 0.0],
    [[0.2003369831, -0.0362207786, 0.592247749, 0.2429277815], 0.0],
    [[0.4160901184, 0.1499094069, 0.4723762574, 0.0047925967], 0.0],
    [[-0.1332716458, 0.1286843714, 0.4578311077, -0.0828125646], 0.0],
    [[-0.0965339492, -0.0888721601, 0.0674008359, 0.273107275], 0.0],
    [[1.1446857578, 0.2068841463, 0.5627325617, 0.6588644044], 0.0],
    [[0.1246586779, 0.89198765, 0.4765024435, 0.6275846291], 0.0],
    [[834.3877172944, 3463.5125318272, 831.5751221024, 3463.716549207], 0],
    [[425.2914915577, 7.2534970734, 410.2249621772, 6.8729185236], 0.0],
    [[-0.0773061666, -0.167516406, 0.0543271937, 0.3031860352], 0.0],
    [[0.0357049222, -0.0334898736, -0.114429559, 0.2997972207], 0.0],
    [[0.1662754607, 0.0341925282, -0.0512390273, 0.0638606023], 0.0],
    [[1.1388856093, 0.2637333896, 0.2828034977, -0.1763537536], 0.0],
    [[518.6173344035, 3365.1675824212, 538.3343852509, 3365.3301038666], 0.0],
    [[0.6755318834, 0.1819658377, 0.6509834012, 0.5053309105], 0.0],
    [[0.2342829686, -0.1037659124, 0.6758371005, 0.4908106857], 0.0],
    [[0.1048324952, -0.0508065369, 0.243878566, 0.1289279044], 0.0],
    [[0.1798745306, 0.0322275749, 0.6106355075, 0.0397609247], 0.0],
    [[0.6101088358, 0.1981202964, 0.0932446752, -0.0776486692], 0.0],
    [[0.0056529164, 0.0999808228, 0.1380425319, -0.1075664863], 0.0],
    [[0.4956879021, 0.2861726139, -0.0003620125, -0.1427854548], 0.0],
    [[0.2003369831, -0.0362207786, 0.592247749, 0.2429277815], 0.0],
    [[0.4160901184, 0.1499094069, 0.4723762574, 0.0047925967], 0.0],
    [[-0.1332716458, 0.1286843714, 0.4578311077, -0.0828125646], 0.0],
    [[-0.0965339492, -0.0888721601, 0.0674008359, 0.273107275], 0.0],
    [[1.1446857578, 0.2068841463, 0.5627325617, 0.6588644044], 0.0],
    [[0.1246586779, 0.89198765, 0.4765024435, 0.6275846291], 0.0],
    [[0.3877172944, 3463.5125318272, 831.5751221024, 3463.716549207], 0],
    [[425.2914915577, 7.2534970734, 410.2249621772, 6.8729185236], 0.0],
    [[-0.0773061666, -0.167516406, 0.0543271937, 0.3031860352], 0.0],
    [[0.0357049222, -0.0334898736, -0.114429559, 0.2997972207], 0.0],
    [[1.1388856093, 0.2637333896, 0.2828034977, -0.1763537536], 0.0]
]

neuroniosSaida = [
    [[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0],
    [[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0],
    [[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0]
]

#o nosso bias
bias = -0.3
biasOculto = -0.3
biasSaida = -0.3

#a nossa taxa de aprendizado
aprendizado = 0.28

#contador de eras
e = 0

#contador de acertos da era
c = 0

#contador de acertos global
g = []

#contador de amplitude
a = 0

dataset = handleIris('/mnt/usb-Generic_STORAGE_DEVICE_000000000819-0:0-part1/documentos/perceptrons/MLP/iris.data')
#shuffle(dataset)

#setando os casos a serem apresentados para a rede no treinamento
data = dataset[0:37]

for l in range(50, 87):
    data.append(dataset[l])

for l in range(100, 137):
    data.append(dataset[l])

shuffle(data)

#setando os casos para o teste da rede
test = dataset[37:50]

for l in range(86, 99):
    test.append(dataset[l])

for l in range(137, 150):
    test.append(dataset[l])


for c in range(0, len(neuroniosEntrada)):
    for i in range(0, len(neuroniosEntrada[c][0])):
        neuroniosEntrada[c][0][i] = (randint(0,79))/100

for c in range(0, len(neuronioOculto)):
    for i in range(20, len(neuronioOculto[c][0])):
        neuronioOculto[c][0][i] = (randint(0,79))/100

for c in range(0, len(neuroniosSaida)):
    for i in range(0, len(neuroniosSaida[c][0])):
        neuroniosSaida[c][0][i] = (randint(0,79))/100

while(e <= 400):    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#seletor de entradas
    for planta in data:

        #resultado esperado da camada
        esperado = planta[-1]
        
        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #calculo da camada oculta 0

        #calculo da camada
        somCamadaOculta0 = perceptron(neuroniosEntrada, planta[0:-1], bias)

        #pega a resposta da camada oculta 0
        respCamadaOculta0 = respostaCamada(neuroniosEntrada)

        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #calculo da camada oculta
        
        #calculo da camada
        somCamadaOculta = perceptron(neuronioOculto, respCamadaOculta0, biasOculto)

        #pega as respostas da camada oculta
        respCamadaOculta = respostaCamada(neuronioOculto)

        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #calculo da camada de saída

        somCamadaSaida = perceptron(neuroniosSaida, respCamadaOculta, biasSaida)

        respCamadaSaida = respostaCamada(neuroniosSaida)

        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #resultados
        c += resultados(respCamadaSaida, esperado)
        
        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #treinamento
        
        #camada de saída
        erroSaida = []

        erroSaida = gradienteErroSaida(respCamadaSaida, esperado)

        #camada oculta
        erroOculto = []
        for neuronio in range(0,len(neuronioOculto)):
            erroOculto.append(gradienteErroOculto(neuroniosSaida, erroSaida, neuronio, somCamadaOculta))

        #camada entrada
        erroEntrada = []
        for neuronio in range(0, len(somCamadaOculta0)):
            erroEntrada.append(gradienteErroOculto(neuronioOculto, erroOculto, neuronio, somCamadaOculta0))
    
        correcaoErro(neuroniosSaida, aprendizado, erroSaida, respCamadaOculta)
        correcaoErro(neuronioOculto, aprendizado, erroOculto, respCamadaOculta0)
        correcaoErro(neuroniosEntrada, aprendizado, erroEntrada, planta[0:-1])

        #bias
        """ bias = correcaoBias(bias, respCamadaSaida, esperado)
        biasOculto = correcaoBias(biasOculto, respCamadaSaida, esperado)
        biasSaida = correcaoBias(biasSaida, respCamadaSaida, esperado) """

    print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-')
    print(f'O número de acertos da era foi: {c}')
    #sleep(0.5)
    g.append(c)
    c = 0
    e += 1
    a += 1
    if(e > 20 and a>20):
        #se a amplitude do ruido for maior ou igual à 15
        if(max(g[-20:]) - min(g[-20:]) >= 7):
            if(aprendizado > 0.14):
                aprendizado -= 0.04
            elif(aprendizado < 0.03):
                aprendizado = 0.01
            else:
                aprendizado -= 0.02
            a = 0
    shuffle(data)
print(f'o maior número de acertos dos treinamentos foi: {max(g)} na era {g.index(max(g))}')
for i in g:
    print('.'*i, '\n')

c = 0
g = []
#casos fora do teste
for planta in test:
    #resultado esperado da camada
    esperado = planta[-1]
    
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #calculo da camada oculta 0

    #calculo da camada
    somCamadaOculta0 = perceptron(neuroniosEntrada, planta[0:-1], bias)

    #pega a resposta da camada oculta 0
    respCamadaOculta0 = respostaCamada(neuroniosEntrada)

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #calculo da camada oculta
    
    #calculo da camada
    somCamadaOculta = perceptron(neuronioOculto, respCamadaOculta0, biasOculto)

    #pega as respostas da camada oculta
    respCamadaOculta = respostaCamada(neuronioOculto)

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #calculo da camada de saída

    somCamadaSaida = perceptron(neuroniosSaida, respCamadaOculta, biasSaida)

    respCamadaSaida = respostaCamada(neuroniosSaida)

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #resultados
    c += resultados(respCamadaSaida, esperado)
print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-')
print(f'O número de acertos dos casos forado teste foi: {c} de {len(test)}')