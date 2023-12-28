""" 
    vamos tentar melhorar o jogo anterior com os novos conhecimentos que adquirimos
"""

#import necessários para o jogo
from random import randint
from time import sleep
from os import system
from copy import deepcopy

#import necessários para a rede neural
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#funções e classes da rede neural
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(48,100)
        #self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(100, 193)
        #self.fc3 = nn.Linear(193, 80)
        self.out = nn.Linear(193, 3)
    
    def forward(self, entradas):
        """ 
            método responsável por fazer a propagação da rede, e gerar a previsão dela.
            :param entradas: as entradas da rede neural
            :type entradas: tensor
            :return: tensor 
          
        """
        x = entradas
        #print(len(x))
        x = F.softsign(self.fc1(x))
        x = F.softmax(self.fc2(x))
        #x = F.softsign(self.fc3(x))
        #x = self.dropout(x)
        resposta = F.tanh(self.out(x))

        return resposta

def resultados(respCamadaSaida):

    if respCamadaSaida[0] > respCamadaSaida[1] and respCamadaSaida[0] > respCamadaSaida[2]:
        return 1
    elif respCamadaSaida[1] > respCamadaSaida[0] and respCamadaSaida[1] > respCamadaSaida[2]:
        return 0
    else:
        return -1

def whatShouldIdo(enemies, playerPos, hist):

    if hist*10 > 3:
        hist += 0.05
    
    answer = torch.tensor([0, 0, 0]).float()
    #linha necessária para que possamos saber qual é o inimigo mais próximo do player
    enemiesPos = sorted(enemies, key=lambda x: x[1], reverse=True)

    #atirar caso o inimigo esteja a vista
    if enemiesPos[0][0] == playerPos:
            answer += torch.tensor([1*(1-hist), 0, 0]).float()
            return answer
    
    for enemy in range (0, len(enemiesPos)):
        if enemiesPos[enemy][0] == playerPos:
            answer += torch.tensor([0.2*(1-hist), 0, 0]).float()

    #andar para buscar o inimigo, com a condição de esperar caso ele esteja na próxima
    if(enemiesPos[0][2] == 0):
        hist = 0
        if(enemies[0][0]-1 == playerPos):
            #viés de parar
            if(enemiesPosition[0][0] < playerPos):
                answer += torch.tensor([0.5, 0, 1]).float()
                return answer
            else:
                answer += torch.tensor([0.5, 1, 0]).float()
                return answer
        else:
            #viés de andar
            if(enemiesPosition[0][0] < playerPos):
                answer += torch.tensor([0.25, 0, 1]).float()
                return answer
            else:
                answer += torch.tensor([0.25, 1, 0]).float()
                return answer
    else:
        hist=0
        if(enemies[0][0]+1 == playerPos):
            #viés de parar
            if(enemiesPosition[0][0] > playerPos):
                answer += torch.tensor([0.5, 1, 0]).float()
                return answer
            else:
                answer += torch.tensor([0.5, 0, 1]).float()
                return answer
        else:
            #vies de andar
            if(enemiesPosition[0][0] > playerPos):
                answer += torch.tensor([0.25, 1, 0]).float()
                return answer
            else:
                answer += torch.tensor([0.25, 0, 1]).float()
                return answer

def waybackMachine(timeMap):
    past = (np.array(timeMap[0]) - np.array([
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46]]))
    
    present = np.array(timeMap[1]) - np.array([
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46],
                                                [46, 46, 46, 46, 46, 46, 46, 46]])
    view = present - past
    return torch.from_numpy(view.flatten()).float()

#funções e classes do jogo
def insertEnemy(x, y, worldMap):
    worldMap[y][x] = 72

def insertPlayer(x, worldMap):
    worldMap[5][x] = 94

def insertShoot(x, y, worldMap):
    worldMap[y][x] = 42

def delete(x, y, worldMap):
    worldMap[y][x] = 46

def setEnemies(n, enemiesPosition, worldMap):
    """ 
        adciona nossos inimigos na grade do worldMapa
    """
    for count in range(0, n):
        enemyPosX = randint(0, 7)
        enemyPosY = randint(0, 2)
        enemiesPosition.append([enemyPosX, enemyPosY, 0])
        insertEnemy(enemyPosX, enemyPosY, worldMap)
    
    #linha necessária para que na classe shoot nós não acertemos um bicho que esta atrás ao invés do que está na frente
    enemiesPosition = sorted(enemiesPosition, key=lambda x: x[1], reverse=True)

def setPlayer(playerPos, worldMap):
    insertPlayer(playerPos, worldMap)

def moveEnemies(enemiesPosition, worldMap):
    """ 
        movimenta o inimigo pelo mapa
    """

    #para cada inimigo presente no mapa
    for enemy in range(0, len(enemiesPosition)):

        if(enemiesPosition[enemy][1] == 5):
            print('GAME OVER!!')
            return 1
        #se ele alcançar a borda
        if((enemiesPosition[enemy][0] == 7 and enemiesPosition[enemy][2] == 1) or (enemiesPosition[enemy][0] == 0 and enemiesPosition[enemy][2] == 0)):
            #apaga o inimigo
            delete(enemiesPosition[enemy][0], enemiesPosition[enemy][1], worldMap)

            #atualiza sua posição no mapa
            enemiesPosition[enemy][1] += 1
            
            #muda a posição de andar para a nova direção
            if enemiesPosition[enemy][2] == 0 :
                enemiesPosition[enemy][2] = 1
            else:
                enemiesPosition[enemy][2] = 0

            #recria o inimigo na tela
            insertEnemy(enemiesPosition[enemy][0], enemiesPosition[enemy][1], worldMap)

        else:
            delete(enemiesPosition[enemy][0], enemiesPosition[enemy][1], worldMap)
            if enemiesPosition[enemy][2] == 0 :
                if 7 - enemiesPosition[enemy][0] < 6:
                    enemiesPosition[enemy][0] -= randint(1,2)
                else:
                    c = randint(0, 1)
                    if c == 0:
                        enemiesPosition[enemy][0] -= 1

                    #fazer com que a solução de encostar no canto e atirar não funcione
                    else:
                        enemiesPosition[enemy][1] += 1
                        #muda a posição de andar para a nova direção
                        if enemiesPosition[enemy][2] == 0 :
                            enemiesPosition[enemy][2] = 1
                        else:
                            enemiesPosition[enemy][2] = 0
            else:
                if 7 - enemiesPosition[enemy][0] > 2:
                    enemiesPosition[enemy][0] += randint(1,2)
                else:
                    c = randint(0, 1)
                    if c == 0:
                        enemiesPosition[enemy][0] += 1

                    #fazer com que a solução de encostar no canto e atirar não funcione
                    else:
                        enemiesPosition[enemy][1] += 1
                        #muda a posição de andar para a nova direção
                        if enemiesPosition[enemy][2] == 0 :
                            enemiesPosition[enemy][2] = 1
                        else:
                            enemiesPosition[enemy][2] = 0

            insertEnemy(enemiesPosition[enemy][0], enemiesPosition[enemy][1], worldMap)

def shootAnimation(x, end, worldMap):
    for count in range(4, end-1, -1):
        sleep(0.2)
        insertShoot(x, count, worldMap)
        showMap(worldMap)
        delete(x, count, worldMap)
    showMap(worldMap)

def shoot(enemiesPos, shootPos, worldMap):
    flag = True
    for enemy in range (0, len(enemiesPos)):
        if enemiesPos[enemy][0] == shootPos:
            flag = False
            shootAnimation(shootPos, enemiesPos[enemy][1], worldMap=worldMap)

            #deleta o inimigo acertado pelo tiro
            enemiesPos.pop(enemy)

            #cria outro inimigo para o jogo continuar
            setEnemies(1, enemiesPos, worldMap)

    if flag:
        shootAnimation(shootPos, 0, worldMap)

def playerAction(command, playerPos, wordmap):

    #atirar
    if command == 1:
        shoot(enemiesPosition, playerPos, wordmap)

    #mover para a direita
    elif command == 0:
        if playerPos < 7:
            delete(playerPos, 5, wordmap)
            playerPos += 1
            insertPlayer(playerPos, worldMap)
    #mover para a esquerda
    elif command == -1:
        if playerPos > 0:
            delete(playerPos, 5, wordmap)
            playerPos -= 1
            insertPlayer(playerPos, worldMap)
    else:
        pass
    return playerPos

def showMap(worldMap, out=''):
    """ 
        exibe o mapa na tela transformando seus valores com a tabela ascii 
    """
    system('clear')
    for line in worldMap:
        for col in line:
            print(chr(col), end=' ')
        print('')
    print(out)


#variáveis globais
worldMap = []
enemiesPosition = []
playerPos = 3

#instancia da rede neural
net = Net()
#o calculo do nosso gradiente de erro utilizará este crossentropyLoss pois queremos calcular o erro de "cada classe"
gradienteErro = nn.CrossEntropyLoss()
#equivale a nossa função de correção de erro, onde usavamos a regra delta, agora vamos utilizar um optimizer
correcaoErro = optim.SGD(net.parameters(), lr=0.18, momentum=0.15, nesterov=True) #optim.Adam(net.parameters(), lr=0.1)
#seta a parte de treinamento ou de jogo da máquina
teacher = True
timeMap = []
derrotas = 0
hist = 0

#loop para a IA não perder e ter que começar do 0
while True:
    worldMap = [
        [46, 46, 46, 46, 46, 46, 46, 46],
        [46, 46, 46, 46, 46, 46, 46, 46],
        [46, 46, 46, 46, 46, 46, 46, 46],
        [46, 46, 46, 46, 46, 46, 46, 46],
        [46, 46, 46, 46, 46, 46, 46, 46],
        [46, 46, 46, 46, 46, 46, 46, 46]
    ]
    enemiesPosition = []
    playerPos = 3
    #setando o mapa do jogo
    setEnemies(4, enemiesPosition, worldMap)
    setPlayer(playerPos, worldMap)
    print(f"\033[1;31m NÚMERO DE DERROTAS: \033[1;33m {derrotas} \033[0m")
    sleep(1)
    #loop do jogo
    while True:

        view = 0
        if len(timeMap) >= 2:
            view = waybackMachine(timeMap)
        else:
            timeMap.append(deepcopy(worldMap))
            view = timeMap[0] - np.array([
                                            [46, 46, 46, 46, 46, 46, 46, 46],
                                            [46, 46, 46, 46, 46, 46, 46, 46],
                                            [46, 46, 46, 46, 46, 46, 46, 46],
                                            [46, 46, 46, 46, 46, 46, 46, 46],
                                            [46, 46, 46, 46, 46, 46, 46, 46],
                                            [46, 46, 46, 46, 46, 46, 46, 46]])
            view = torch.from_numpy(view.flatten()).float()

        #passa o mapa para a rede neural fazer a jogada
        out = net(view)

        #exibe o mapa na tela
        showMap(worldMap, out)

        #calcula a saída esperada da rede
        expected = whatShouldIdo(enemiesPosition, playerPos, hist)

        #"interpreta" a 1ª jogada da rede neural
        command = resultados(out)
        
        #faz a jogada
        playerPos = playerAction(command, playerPos, worldMap)

        #move os NPCs
        kill = moveEnemies(enemiesPosition, worldMap)

        #verifica se o jogo não deu gameOver
        if kill == 1:
            expected = torch.tensor([2, 1.5, 1.5]).float()
            correcaoErro.zero_grad()
            error = gradienteErro(out, expected)
            #realiza o back propagation
            error.backward()

            #corrige o erro
            correcaoErro.step()
            kill = 0
            break

        #treina a rede neural
        if teacher:
            correcaoErro.zero_grad()
            error = gradienteErro(out, expected)
            #realiza o back propagation
            error.backward()

            #corrige o erro
            correcaoErro.step()

        #mostra o mapa mudado
        showMap(worldMap, out)

        #gerencia os estados do timeMap
        if len(timeMap) >= 2:
            timeMap[0] = deepcopy(timeMap[1])
            timeMap[1] = deepcopy(worldMap)
        else:
            timeMap.append(deepcopy(worldMap))
        
        #delay para podermos enchergar tudo isso acontecendo
        sleep(0.5)
    
    derrotas += 1