from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.python import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import sys
import random
import numpy as np
import os
import datetime
import time

def make_card_data():
    with open('cards.txt') as f:
        cardnamesrawunsplit = f.read()
    cardnamesraw = cardnamesrawunsplit.split("\n")
    cardnames = []
    normal = []
    legendary = []
    all = []
    for i in range(0, len(cardnamesraw) - 1):
        index = cardnamesraw.pop(1).replace(" ", "").split(",")

        if(index[0] == ""):
             continue
        c = Card(index)
        cardnames.insert(i, index[0])
        all.append(c)
        if (c.stars == 3):
            normal.append(c)
        elif (c.stars == 5):
            legendary.append(c)
    cardnamesdict = {}
    for i in range(0, len(cardnames)):
        cardnamesdict[cardnames[i]] = i
    return (normal, legendary, all, cardnamesdict)

class TripleTriadBot(object):

    def __init__(self, currentModelID):
        self.testnet = []
        self.currentModelID = currentModelID

        self.normal, self.legendary, self.allcards, self.cardnamedict = make_card_data()
        self.net = []

        for i in range(1):


            model = keras.models.Sequential()

            model.add(keras.layers.Conv2D(30, (1, 9), activation='relu', input_shape=(5, 20, 1), name = "input"))
            model.add(keras.layers.BatchNormalization())

            #model.add(keras.layers.Conv2D(16, (1, 1), activation='relu', input_shape=(5, 20, 1), name = "input"))
            #model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(.5))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(45, activation='softmax'))

            model.add(keras.layers.Dense(45, activation='softmax'))
            model.add(keras.layers.Reshape((1, 45)))

            model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            model.summary()
            self.net.append(model)
            #self.net[i].large_weight_initializer()
        if (currentModelID == -1):
            self.currentModelID = 0
            tf.keras.models.save_model(self.net[0], "ttnn/finalnetworktf{}.txt".format(self.currentModelID))
        self.load()
        for i in range(1):
            self.testnet.append(tf.keras.models.load_model("ttnn/finalnetworktf{}.txt".format(self.currentModelID)))
        random.seed(datetime.datetime.now())

    def getBestNetworkMove(self, outputFromNN, gameboard, shouldBeRandom, training = True):

        reshapedOutput = outputFromNN.reshape(45)
        #print (card.name)
        bestMoves = []
        bestMove = None
        bestConfidence = -1
        for i in range(len(outputFromNN)):
            #translate to cards + positions
            card = None

            if (gameboard.turn % 2 == 0):
                card = gameboard.initialP1Hand[i % 5]
            else:
                card = gameboard.initialP2Hand[i % 5]

            #print (card.name)
            pos = i // 5
            if (gameboard.isValidMove(card, pos)):
                if (bestConfidence < round(reshapedOutput.item(i), 2)):
                    bestConfidence = round(reshapedOutput.item(i), 2)
                    bestMoves = [(card, pos)]
                elif bestConfidence == round(reshapedOutput.item(i), 2):
                    bestMoves.append((card, pos))
        if (len(bestMoves) == 1): bestMove = bestMoves[0]
        if (len(bestMoves) > 1):
            bestMove = bestMoves[random.randint(0, len(bestMoves) - 1)]
        if (shouldBeRandom or len(bestMoves) == 0):
            bestMoves = []
            bestConfidence = 10000

            for i in range(45):
                #translate to cards + positions
                card = None

                if (gameboard.turn % 2 == 0):
                    card = gameboard.initialP1Hand[i % 5]
                else:
                    card = gameboard.initialP2Hand[i % 5]

                #print (card.name)
                pos = i // 5
                if (gameboard.isValidMove(card, pos)):
                    bestMoves.append((card, pos))
            if (len(bestMoves) > 1): bestMove = bestMoves[random.randint(0, len(bestMoves) - 1)]
            else: bestMove = bestMoves[0]
        #print(bestMoves)
        return bestMove

    def generate_random_hand(self):
        firstCard = random.randint(0, len(self.normal) - 1)
        secondCard = random.randint(0, len(self.normal) - 1)
        if (secondCard == firstCard):
            secondCard = (secondCard+1) % len(self.normal)
        thirdCard = random.randint(0, len(self.normal) - 1)
        while (thirdCard == firstCard or thirdCard == secondCard):
            thirdCard = (thirdCard+1) % len(self.normal)
        fourthCard = random.randint(0, len(self.normal) - 1)
        while (fourthCard == firstCard or fourthCard == secondCard or fourthCard == thirdCard):
            fourthCard = (fourthCard+1) % len(self.normal)
        legendary = self.legendary[random.randint(0, len(self.legendary) - 1)]
        lst = [self.normal[firstCard], self.normal[secondCard], self.normal[thirdCard], self.normal[fourthCard], legendary]
        random.shuffle(lst)
        return lst

    def formatGame(self, gamedatafile):

        with open(gamedatafile) as f:
            gamedataunsplit = f.read()
        gamedata = gamedataunsplit.split("\n")

        nplist = [None] * 9
        nplistIn = [None] * 9
        nplistOut = [None] * 9
        print(nplist)
        for i in range(0, len(gamedata), 3):
            #print(i)
            if (gamedata[i] == ""):
                break
            p1Deck = [self.allcards[self.cardnamedict[value]] for value in gamedata[i][len("P1 Deck: "):].replace(" ", "").split("|")]
            p2Deck = [self.allcards[self.cardnamedict[value]] for value in gamedata[i + 1][len("P2 Deck: "):].replace(" ", "").split("|")]
            gameboardData = gamedata[i + 2][len("Cards played: "): -1].replace(" ", "").replace("(Blue)", "").replace("(Red)", "").split("|")
            gameboard = Gameboard(p1Deck, p2Deck)
            for j in range(9):
                if (i + 3 < len(gamedata)):
                    if (("P1W" in gameboardData[-1] or "T" in gameboardData[-1])  and j % 2 == 1):
                        cardName, posstring = gameboardData[j].split(":")
                        pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])
                        gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)
                        continue
                    if (("P1L" in gameboardData[-1] or "T" in gameboardData[-1]) and j % 2 == 0):
                        cardName, posstring = gameboardData[j].split(":")
                        pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])
                        gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)
                        continue
                cardName, posstring = gameboardData[j].split(":")
                pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])

                inputForNN = [0] * 100
                outputForNN = [0] * 9 * 5

                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict)
                if (j % 2 == 0):
                    outputForNN[pos * 5 + p1Deck.index(self.allcards[self.cardnamedict[cardName]])] = 1
                else:
                    outputForNN[pos * 5 + p2Deck.index(self.allcards[self.cardnamedict[cardName]])] = 1



                gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)

                if (nplistIn[0] == None):
                    nplistIn[0] = inputForNN
                    nplistOut[0] = outputForNN
                else:
                    nplistIn[0] += inputForNN
                    nplistOut[0] += outputForNN
        for j in range(1):
            npinput = np.reshape(np.array(nplistIn[j], order='F', ndmin = 4), [(len(nplistIn[j]) // 100), 5, 20, 1])
            npoutput = np.reshape(np.array(nplistOut[j], order='F', ndmin = 4), [(len(nplistOut[j]) // 45), 1, 45])
            nplist[0] = (npinput, npoutput)
        print(npinput.shape)
        return nplist

    def mainmethod(self, iterations = 5, learningrate = 0.2, numgames=100, shouldBeRandom = False, shouldP1Random = False, shouldP2Random = False):

        for i in range(1000):
            handdata = self.play_games(number_of_games = numgames, outputfile = "ttnn/thousandgames{}.txt".format(i), shouldBeRandom = True, shouldP1Random = shouldP1Random, shouldP2Random = shouldP2Random)
            nplist = self.formatGame("ttnn/thousandgames{}.txt".format(i))

            for j in range(1):
                if shouldP1Random and j % 2 == 0: continue
                if shouldP2Random and j % 2 == 1: continue
                print("Processing network {}".format(j))
                datasetIn, datasetOut = nplist[0]
                self.net[0].fit(datasetIn, datasetOut, epochs=iterations)

                tf.keras.models.save_model(self.net[0], "ttnn/finalnetworktf{}.txt".format(self.currentModelID + 1))
                print("Saved network {} to file!".format(0))
            self.test_network(handdata, number_of_games = numgames)

    def playAgainstBot(self, directory = "nn10/"):

        stringedGamesPlayed = ""
        #testnet = []
        #for i in range(9):
            #testnet.append(network2.load("ttnn/archive/" + directory + "finalnetwork{}.txt".format(i)))
        p1 = self.generate_random_hand()
        p2 = self.generate_random_hand()
        youAreP1 = isP1 = random.randint(0,1)==0


        if youAreP1:
            #p2 = [self.allcards[self.cardnamedict["Cloud"]], self.allcards[self.cardnamedict["Asahi"]], self.allcards[self.cardnamedict["Phoebad"]], self.allcards[self.cardnamedict["Ysayle"]], self.allcards[self.cardnamedict["Hilda"]]]
            random.shuffle(p1)
        else:
            #p1 = [self.allcards[self.cardnamedict["Cloud"]], self.allcards[self.cardnamedict["Asahi"]], self.allcards[self.cardnamedict["Phoebad"]], self.allcards[self.cardnamedict["Ysayle"]], self.allcards[self.cardnamedict["Hilda"]]]
            random.shuffle(p2)
        gameboard = Gameboard(p1, p2)
        for j in range(9):
            inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, reversed_player = True)
            outputFromNN = None
            cardToPlay, posToPlay = (None , 0)
            if j % 2 == 0:
                if youAreP1:
                    shittyInput = True
                    cardNumber = -1
                    pos = -1
                    while (shittyInput):
                        shittyInput = False
                        cardStrings = [None] * 5
                        cardStrings[0] = p1[0].name + "(u: {}, r:{}, d:{}, l:{})".format(p1[0].up, p1[0].right, p1[0].down, p1[0].left)
                        if len(p1) > 1: cardStrings[1] = p1[1].name + "(u: {}, r:{}, d:{}, l:{})".format(p1[1].up, p1[1].right, p1[1].down, p1[1].left)
                        if len(p1) > 2: cardStrings[2] = p1[2].name + "(u: {}, r:{}, d:{}, l:{})".format(p1[2].up, p1[2].right, p1[2].down, p1[2].left)
                        if len(p1) > 3: cardStrings[3] = p1[3].name + "(u: {}, r:{}, d:{}, l:{})".format(p1[3].up, p1[3].right, p1[3].down, p1[3].left)
                        if len(p1) > 4: cardStrings[4] = p1[4].name + "(u: {}, r:{}, d:{}, l:{})".format(p1[4].up, p1[4].right, p1[4].down, p1[4].left)

                        gameboard.printgameboard()
                        print("Your hand is: (1) {}, (2) {}, (3) {}, (4) {}, (5) {}".format(cardStrings[0], cardStrings[1], cardStrings[2], cardStrings[3], cardStrings[4]))
                        try:
                            cardNumber = int(input("What card would you like to play?"))
                            xPos = int(input("X Position? (0 - 2)"))
                            yPos = int(input("Y Position? (0 - 2)"))
                            pos = xPos + yPos * 3
                            if (not gameboard.isValidMove(p1[cardNumber - 1], pos)): shittyInput = True
                        except:
                            shittyInput = True
                    cardToPlay = p1[cardNumber - 1]
                    posToPlay = pos
                    p1.remove(cardToPlay)
                    gameboard.playCard(cardToPlay, posToPlay)
                else:
                    outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                    mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, True, shouldBeRandom = False)
                    print("Odds of bot victory: {}%".format(int(100 * mctsResults[0] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    print("Odds of bot tie: {}%".format(int(100 * mctsResults[1] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    print("Odds of bot loss: {}%".format(int(100 * mctsResults[2] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    gameboard.playCard(cardToPlay, posToPlay)
            else:
                if youAreP1:
                    outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                    mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, False, shouldBeRandom = False)
                    print("Odds of bot victory: {}%".format(int(100 * mctsResults[0] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    print("Odds of bot tie: {}%".format(int(100 * mctsResults[1] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    print("Odds of bot loss: {}%".format(int(100 * mctsResults[2] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    gameboard.playCard(cardToPlay, posToPlay)
                else:
                    shittyInput = True
                    cardNumber = -1
                    pos = -1
                    while (shittyInput):
                        shittyInput = False
                        cardStrings = ["None"] * 5
                        cardStrings[0] = p2[0].name + "(u: {}, r:{}, d:{}, l:{})".format(p2[0].up, p2[0].right, p2[0].down, p2[0].left)
                        if len(p2) > 1: cardStrings[1] = p2[1].name + "(u: {}, r:{}, d:{}, l:{})".format(p2[1].up, p2[1].right, p2[1].down, p2[1].left)
                        if len(p2) > 2: cardStrings[2] = p2[2].name + "(u: {}, r:{}, d:{}, l:{})".format(p2[2].up, p2[2].right, p2[2].down, p2[2].left)
                        if len(p2) > 3: cardStrings[3] = p2[3].name + "(u: {}, r:{}, d:{}, l:{})".format(p2[3].up, p2[3].right, p2[3].down, p2[3].left)
                        if len(p2) > 4: cardStrings[4] = p2[4].name + "(u: {}, r:{}, d:{}, l:{})".format(p2[4].up, p2[4].right, p2[4].down, p2[4].left)

                        gameboard.printgameboard()
                        print("Your hand is: (1) {}, (2) {}, (3) {}, (4) {}, (5) {}".format(cardStrings[0], cardStrings[1], cardStrings[2], cardStrings[3], cardStrings[4]))
                        try:
                            cardNumber = int(input("What card would you like to play?"))
                            xPos = int(input("X Position? (0, 1, 2)"))
                            yPos = int(input("Y Position? (0, 1, 2)"))
                            pos = xPos + yPos * 3
                            if (not gameboard.isValidMove(p2[cardNumber - 1], pos)): shittyInput = True
                        except:
                            shittyInput = True
                    cardToPlay = p2[cardNumber - 1]
                    posToPlay = pos
                    p2.remove(cardToPlay)
                    gameboard.playCard(cardToPlay, posToPlay)
        print("Final Results: ")
        gameboard.printgameboard()
        if (gameboard.score > 0): print("P1 Wins!")
        if (gameboard.score == 0): print("Tie!")
        if (gameboard.score < 0): print("P2 Wins!")



    def MCTS(self, gameboard, turnsToCheck, isP1, training = False, shouldBeRandom = False, topLayer = True, layerOutput = None):
        startTime = int(round(time.time() * 1000))
        ##random?
        if shouldBeRandom:

            for k in range(45):

                topChoiceIndex = k
                card = None

                if (gameboard.turn % 2 == 0):
                    card = gameboard.initialP1Hand[topChoiceIndex % 5]
                else:
                    card = gameboard.initialP2Hand[topChoiceIndex % 5]


                pos = topChoiceIndex // 5

                if (gameboard.isValidMove(card, pos)):
                    return ([], card, pos)

        #special conditions?
        if(turnsToCheck <= 0 or gameboard.turn >= 9):
            if isP1:
                #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))
                if gameboard.score > 0: return ([1, 0, 0], None, -1)
                elif gameboard.score == 0: return ([0, 1, 0], None, -1)
                else: return ([0, 0, 1], None,-1)
            else:
                #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))
                if gameboard.score < 0: return ([1, 0, 0], None,-1)
                elif gameboard.score == 0: return ([0, 1, 0], None,-1)
                else: return ([0, 0, 1], None,-1)




        outputFromNN = None
        if layerOutput is not None:
            outputFromNN = layerOutput
        else:
            inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, reversed_player = (isP1 or gameboard.turn % 2 == 1) and not ((isP1 and gameboard.turn % 2 == 1)))
            outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))

        topChoiceIndexes =[]
        desiredValidIndexes = 6
        scores = [[0, 0, 0]] * desiredValidIndexes
        i1 = 0
        validIndexesFound = 0
        validIndexes = []
        b = np.reshape(outputFromNN, (45)).tolist()
        if (training):
            for i in range(45): b[i] += (random.random() * 0.1 - 0.05)
        indexesInOrder = {}
        sumScores = [0, 0, 0]
        usedIndexes = [False] * 45
        for i in range(45):
            bestIndex = 7000
            bestIndexValue = -100000
            for j in range(45):
                if (b[j] >= bestIndexValue and not usedIndexes[j]):
                    bestIndex = j
                    bestIndexValue = b[j]
            usedIndexes[bestIndex] = True
            indexesInOrder[i] = bestIndex
        while (len(validIndexes) < desiredValidIndexes and i1 < 45):

            topChoiceIndex = indexesInOrder[i1]
            card = None

            if (gameboard.turn % 2 == 0):
                card = gameboard.initialP1Hand[topChoiceIndex % 5]
            else:
                card = gameboard.initialP2Hand[topChoiceIndex % 5]


            pos = topChoiceIndex // 5

            if (not gameboard.isValidMove(card, pos)):
                i1+=1
                continue


            validIndexes.append(topChoiceIndex)
            i1+=1
        inputsForNN = []
        gbCopies = []

        for i in range(len(validIndexes)):
            redhand = [None]
            card = None

            if (gameboard.turn % 2 == 0):
                card = gameboard.initialP1Hand[validIndexes[i] % 5]
            else:
                card = gameboard.initialP2Hand[validIndexes[i] % 5]
            pos = validIndexes[i] // 5
            if validIndexes[i] % 5 >= 3:
                if (isP1 or gameboard.turn % 2 == 0) and not ((isP1 and gameboard.turn % 2 == 0)):
                    if gameboard.turn % 2 == 1:
                        redhand = [value for value in gameboard.initialP2Hand[0:3] if value in gameboard.p2Hand]
                        #print(len(bluehand))
                        redhand = gameboard.getExtraCardsForHand(redhand, self.allcards, self.normal, self.legendary, self.cardnamedict)
                        if len(redhand) > 1:
                            card = redhand[validIndexes[i] % 5 - 5]
                        else:
                            card = redhand[0]
                    else:
                        redhand = [value for value in gameboard.initialP1Hand[0:3] if value in gameboard.p1Hand]
                        #print(len(redhand))
                        #print(len(redhand))
                        redhand = gameboard.getExtraCardsForHand(redhand, self.allcards, self.normal, self.legendary, self.cardnamedict)

                        if len(redhand) > 1:
                            card = redhand[validIndexes[i] % 5 - 5]
                        else:
                            card = redhand[0]
            gbCopy = gameboard.clone()
            gbCopy.playCard(card, pos)

            gbCopies.append(gbCopy)
        inputsForNN = []
        for i in range(len(validIndexes)):
            inputsForNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, reversed_player = (isP1 or gameboard.turn % 2 == 1) and not ((isP1 and gameboard.turn % 2 == 1)))

        outputsFromNN = np.reshape(self.net[0].predict(np.reshape(np.array(inputsForNN, order='F', ndmin = 4), (-1, 5, 20, 1))), (len(validIndexes), 45))
        for i in range(len(validIndexes)):
            mctsdata, card, pos = self.MCTS(gbCopies[i], turnsToCheck - 1, isP1, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, layerOutput = outputsFromNN[i])
            scores[i] = mctsdata

        bestScore = -1000
        bestIndex = -1
        scoreIndex = -70
        for i in range(len(validIndexes)):
            indexScore = scores[i][0] - scores[i][2] - 0.01 * scores[i][1]
            if (bestScore < indexScore):
                bestIndex = validIndexes[i]
                bestScore = indexScore
                scoreIndex = i


        for i in range(len(validIndexes)):
            sumScores[0] += scores[i][0]
            sumScores[1] += scores[i][1]
            sumScores[2] += scores[i][2]
        if (bestIndex == -1):
            for i in range(45):
                if (gameboard.turn % 2 == 0):
                    card = gameboard.initialP1Hand[i % 5]
                else:
                    card = gameboard.initialP2Hand[i % 5]


                pos = i // 5

                if (gameboard.isValidMove(card, pos)):
                    bestIndex = i
                    break

        card = None

        if (gameboard.turn % 2 == 0):
            card = gameboard.initialP1Hand[bestIndex % 5]
        else:
            card = gameboard.initialP2Hand[bestIndex % 5]

        #print(sumScores)
        #print(indexesInOrder)
        #print(b)
        pos = bestIndex // 5
        #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))
        if (layerOutput is None): return (scores[scoreIndex], card, pos)
        return (sumScores, card, pos)

    def test_network(self, handdata, directory = "nn1/", number_of_games = 1000):
        stringedGamesPlayed = ""

        winsAsP1 = 0
        winsAsP2 = 0
        tiesAsP1 = 0
        tiesAsP2 = 0
        lossesAsP1 = 0
        lossesAsP2 = 0
        for i in range(number_of_games):
            #p1, p2, score = handdata[i]
            #isP1 = score > 0
            #if (score == 0): continue
            p1 = self.generate_random_hand()
            p2 = self.generate_random_hand()
            isP1 = random.randint(0,1)==0

            gameboard = Gameboard(p1.copy(), p2.copy())
            for j in range(9):
                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict)
                outputFromNN = None
                cardToPlay, posToPlay = (None, 0)
                if j % 2 == 0:
                    if isP1:
                        outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, True, shouldBeRandom = False)
                    else:
                        outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, True, shouldBeRandom = False)

                else:
                    if isP1:
                        outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, False, shouldBeRandom = False)

                    else:
                        outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, False, shouldBeRandom = False)

                #print(outputFromNN.shape)
                gameboard.playCard(cardToPlay, posToPlay)
            if isP1:
                if gameboard.score > 0:
                    winsAsP1 +=1
                elif gameboard.score == 0:
                    tiesAsP1 +=1
                else:
                    lossesAsP1 += 1
            else:
                if (gameboard.score < 0):
                    winsAsP2 +=1
                elif gameboard.score == 0:
                    tiesAsP2 +=1
                else:
                    lossesAsP2 += 1
            print(gameboard.toFileString(self.allcards))
        if (winsAsP2 + winsAsP1) / (winsAsP2 + winsAsP1 + lossesAsP1 + lossesAsP2) > 0.535 and (winsAsP1) / (winsAsP1 + lossesAsP1) > 0.435 and (winsAsP2) / (winsAsP2 + lossesAsP2) > 0.435:
            self.currentModelID += 1
            del self.testnet[0]
            self.testnet = [None]
            for i in range(1):
                self.testnet[0] = tf.keras.models.load_model("ttnn/finalnetworktf{}.txt".format(self.currentModelID))
            del self.net[0]
            self.net = [None]
            self.load()
            print("Better network found! New network model is: {}".format(self.currentModelID))
        else:
            del self.testnet[0]
            self.testnet = [None]
            self.testnet[0] = tf.keras.models.load_model("ttnn/finalnetworktf{}.txt".format(self.currentModelID))

            print("Could not find better network!")
        print("{} W / {} T / {} L games against the dummy network as Player 1".format(winsAsP1, tiesAsP1, lossesAsP1))
        print("{} W / {} T / {} L games against the dummy network as Player 2".format(winsAsP2, tiesAsP2, lossesAsP2))

    def outputValueToCardAndPos(self, gameboard, outputValue):
        card = None
        if (gameboard.turn % 2 == 0):
            card = gameboard.initialP1Hand[outputValue % 5]
        else:
            card = gameboard.initialP2Hand[outputValue % 5]

        #print (card.name)
        pos = outputValue // 5
        return (card, pos)

    def play_games(self, number_of_games = 1000, outputfile = "thousandgames.txt", shouldBeRandom = False, shouldP1Random = False, shouldP2Random = False):
        stringedGamesPlayed = ""
        gamedata = []
        random.seed(datetime.datetime.now())
        for i in range(number_of_games):
            isP1 = random.randint(0,1)==0
            startTime = int(round(time.time() * 1000))
            p1 = self.generate_random_hand()
            p2 = self.generate_random_hand()
            gameboard = Gameboard(p1.copy(), p2.copy())
            for j in range(9):
                startTurnTime = int(round(time.time() * 1000))
                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict)
                outputFromNN = None

                #print(outputFromNN.shape)
                cardToPlay, posToPlay = (None, 0)
                if j % 2 == 0:
                    if isP1:
                        outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, True, training = True, shouldBeRandom = False)
                    else:
                        outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, True, training = True, shouldBeRandom = False)
                else:
                    if isP1:
                        outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, False, training = True, shouldBeRandom = False)
                    else:
                        outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 20, 1)))
                        mctsResults, cardToPlay, posToPlay = self.MCTS(gameboard.clone(), 4, False, training = True, shouldBeRandom = False)
                gameboard.playCard(cardToPlay, posToPlay)
                #print ("Time to complete a turn: {} ms".format(i, int(round(time.time() * 1000)) - startTurnTime))
            gamedata.append((p1, p2, gameboard.score))
            #print ("Time to complete a game: {} ms".format(i, int(round(time.time() * 1000)) - startTime))
            stringedGamesPlayed += gameboard.toFileString(self.allcards) + "\n"
            if i % (number_of_games // 100) == 0:
                print("Played {}% of games".format(i / number_of_games * 100))
        with open(outputfile, "w") as f:
            f.write(stringedGamesPlayed)
        return gamedata

    def load(self):
        self.net = [None]
        for i in range(1):
            self.net[0] = tf.keras.models.load_model("ttnn/finalnetworktf{}.txt".format(self.currentModelID))
class Card(object):
    def __init__(self, carddata):
        self.name = carddata[0]
        self.stars = int(carddata[1])
        self.left = int(carddata[2])
        self.up = int(carddata[3])
        self.right = int(carddata[4])
        self.down = int(carddata[5])


class Gameboard(object):

    def __init__(self, p1, p2):
        self.p1Hand = [p1[0], p1[1], p1[2], p1[3], p1[4]]
        self.p1Hidden = [False] * 5
        self.p1Hidden[random.randint(0, 4)] = True
        otherInt = random.randint(0, 4) #this is why we dont name stuff at midnight
        while (self.p1Hidden[otherInt]):
            otherInt = random.randint(0, 4)
        self.p1Hidden[otherInt]
        self.p2Hand = [p2[0], p2[1], p2[2], p2[3], p2[4]]
        self.p2Hidden = [False] * 5
        self.p2Hidden[random.randint(0, 4)] = True
        while (self.p1Hidden[otherInt]):
            otherInt = random.randint(0, 4)
        self.p2Hidden[otherInt]
        self.initialP1Hand = [p1[0], p1[1], p1[2], p1[3], p1[4]]
        self.initialP2Hand = [p2[0], p2[1], p2[2], p2[3], p2[4]]
        self.gameboard = [None] * 9
        self.turn = 0
        self.score = 0

    def clone(self):
        clone = Gameboard(self.initialP1Hand, self.initialP2Hand)
        clone.p1Hand = self.initialP1Hand.copy()
        clone.p1Hand = self.initialP2Hand.copy()
        clone.p1Hand = self.p1Hand.copy()
        clone.p2Hand = self.p2Hand.copy()
        for i in range (9):
            if (self.gameboard[i] is not None):
                clone.gameboard[i] = self.gameboard[i].clone()
        clone.turn = self.turn
        clone.score = self.score
        return clone

    def getExtraCardsForHand(self, hand, allcards, normalcards, legendarycards, cardnamedict):
        contains5Star = False
        for i in range(len(hand)):
            if hand[i].stars == 5:
                contains5Star = True
                break
        fiveStarWasPlayed = False
        initialHandLength = 0
        secretUnplayedCards = 0
        if self.turn % 2 == 0:
            initialHandLength = len(self.p2Hand)
            secretUnplayedCards = len([value for value in self.initialP2Hand[3:5] if value in self.p2Hand])
            for i in self.initialP2Hand[0:3]:
                if i.stars == 5:
                    fiveStarWasPlayed = (i in hand)
                    break


        else:
            initialHandLength = len(self.p1Hand)
            secretUnplayedCards = len([value for value in self.initialP1Hand[3:5] if value in self.p1Hand])
            for i in self.initialP1Hand[0:3]:
                if i.stars == 5:
                    fiveStarWasPlayed = (i in hand)
                    break

        normalUnplayedCards = len(hand)
        totalUnplayedCards = normalUnplayedCards + secretUnplayedCards
        if (contains5Star or fiveStarWasPlayed):
            while(len(hand) < totalUnplayedCards):
                hand.append(Card(["Three Star Wild", 3, 8, 8, 8 , 8]))
        else:
            hand.append(Card(["Five Star Wild", 3, 10, 10, 10 ,10]))
            while(len(hand) < totalUnplayedCards):
                hand.append(Card(["Three Star Wild", 3, 8, 8, 8 , 8]))
        return hand


    def getGameboardInputArray(self, allcards, normalcards, legendarycards, cardnamedict, reversed_player = False):
        redhand = self.p1Hand #dont actually call them this! fix at some point
        bluehand = self.p2Hand

        if (self.turn % 2 == 0):
            bluehand = [value for value in self.initialP2Hand[0:3] if value in self.p2Hand]
            #print(len(bluehand))
            if reversed_player:
                bluehand = self.getExtraCardsForHand(bluehand, allcards, normalcards, legendarycards, cardnamedict)
                #print(len(bluehand))


        else:
            redhand = [value for value in self.initialP1Hand[0:3] if value in self.p1Hand]
            #print(len(redhand))
            if reversed_player:
                redhand = self.getExtraCardsForHand(redhand, allcards, normalcards, legendarycards, cardnamedict)
                #print(len(redhand))
#


        input = [0] * 100 #len(cardnames) * 8 + len(cardnames) * 9 * 9
        #inputs for gameboardstate
        #inputs are sequentially after the 8 known hand cards
        #
        for k in range(9):
            square = self.gameboard[k]
            if (square is not None):
                #print(square.card.name)
                arrayIndex = 5 * 10 + 5 * square.pos
                input[arrayIndex + 0] = square.card.left / 10
                input[arrayIndex + 1] = square.card.up / 10
                input[arrayIndex + 2] = square.card.right / 10
                input[arrayIndex + 3] = square.card.down / 10
                input[arrayIndex + 4] = square.owner
        #inputs for known hand cards
        try:
            for k in range(5):
                if (k < 5):
                    if (self.initialP1Hand[k] in redhand):
                        input[5 * k + 0] = self.initialP1Hand[k].left / 10
                        input[5 * k + 1] = self.initialP1Hand[k].up / 10
                        input[5 * k + 2] = self.initialP1Hand[k].right / 10
                        input[5 * k + 3] = self.initialP1Hand[k].down / 10
                        input[5 * k + 4] = 0
            if (reversed_player and self.turn % 2 == 1):
                secretUnplayedCards = len([value for value in self.initialP1Hand[3:5] if value in self.p1Hand])
                #print(len(redhand))
                if secretUnplayedCards == 2:

                    input[5 * (3 + 0) + 0] = redhand[len(redhand)-2].left / 10
                    input[5 * (3 + 0) + 1] = redhand[len(redhand)-2].up / 10
                    input[5 * (3 + 0) + 2] = redhand[len(redhand)-2].right / 10
                    input[5 * (3 + 0) + 3] = redhand[len(redhand)-2].down / 10
                    input[5 * (3 + 0) + 4] = 0
                if secretUnplayedCards >= 1:
                    input[5 * (4 + 0) + 0] = redhand[len(redhand)-1].left / 10
                    input[5 * (4 + 0) + 1] = redhand[len(redhand)-1].up / 10
                    input[5 * (4 + 0) + 2] = redhand[len(redhand)-1].right / 10
                    input[5 * (4 + 0) + 3] = redhand[len(redhand)-1].down / 10
                    input[5 * (4 + 0) + 4] = 0

        except TypeError:
            #nothing needs to be done
            redhand = redhand
        try:
            for k in range(5):
                if (k < 5):
                    if (self.initialP2Hand[k] in bluehand):
                        input[5 * (k + 5) + 0] = self.initialP2Hand[k].left / 10
                        input[5 * (k + 5) + 1] = self.initialP2Hand[k].up / 10
                        input[5 * (k + 5) + 2] = self.initialP2Hand[k].right / 10
                        input[5 * (k + 5) + 3] = self.initialP2Hand[k].down / 10
                        input[5 * (k + 5) + 4] = 1
            if (reversed_player and self.turn % 2 == 0):
                secretUnplayedCards = len([value for value in self.initialP2Hand[3:] if value in self.p2Hand])
                #print(len(bluehand))
                if secretUnplayedCards == 2:

                    input[5 * (3 + 5) + 0] = bluehand[len(bluehand)-2].left / 10
                    input[5 * (3 + 5) + 1] = bluehand[len(bluehand)-2].up / 10
                    input[5 * (3 + 5) + 2] = bluehand[len(bluehand)-2].right / 10
                    input[5 * (3 + 5) + 3] = bluehand[len(bluehand)-2].down / 10
                    input[5 * (3 + 5) + 4] = 1
                if secretUnplayedCards >= 1:
                    input[5 * (4 + 5) + 0] = bluehand[len(bluehand)-1].left / 10
                    input[5 * (4 + 5) + 1] = bluehand[len(bluehand)-1].up / 10
                    input[5 * (4 + 5) + 2] = bluehand[len(bluehand)-1].right / 10
                    input[5 * (4 + 5) + 3] = bluehand[len(bluehand)-1].down / 10
                    input[5 * (4 + 5) + 4] = 1
        except TypeError:
            bluehand = bluehand
        return input

    def isValidMove(self, card, position):


        if (self.gameboard[position] is not None):
            return False;
        if (self.turn % 2 == 0):
            return card in self.p1Hand
        else:
            return card in self.p2Hand

    def printgameboard(self):
        sb = ""
        sb += ("    0     1     2\n")
        sb += (" -------------------\n")
        cs = [[" "] * 4] * 9
        os = [" "] * 9
        for i in range(9):
            if (self.gameboard[i] is not None):
                placeholder = [" "] * 4
                if (self.gameboard[i].card.up != 10): placeholder[0] = str(self.gameboard[i].card.up)
                else: placeholder[0] = "A"
                if (self.gameboard[i].card.right != 10): placeholder[1] = str(self.gameboard[i].card.right)
                else: placeholder[1] = "A"
                if (self.gameboard[i].card.down != 10): placeholder[2] = str(self.gameboard[i].card.down)
                else: placeholder[2] = "A"
                if (self.gameboard[i].card.left != 10): placeholder[3] =  str(self.gameboard[i].card.left)
                else: placeholder[3] = "A"
                cs[self.gameboard[i].pos] = placeholder
                os[self.gameboard[i].pos] = self.gameboard[i].owner + 1


        for i in range(0, 9, 3):
            sb += " |  {}  |  {}  |  {}  |\n".format(cs[i][0], cs[i+1][0], cs[i+2][0])
            sb += "{}|{} {} {}|{} {} {}|{} {} {}|\n".format(i // 3, cs[i][3], os[i], cs[i][1], cs[i+1][3], os[i+1], cs[i+1][1], cs[i+2][3], os[i+2], cs[i+2][1])
            sb += " |  {}  |  {}  |  {}  |\n".format(cs[i][2], cs[i+1][2], cs[i+2][2])
            sb += " -------------------\n"
        redhand = []
        if (self.turn % 2 == 0):
            redhand = [value for value in self.initialP2Hand[0:3] if value in self.p2Hand]
            #print(len(self.p2Hand))
        else:
            redhand = [value for value in self.initialP1Hand[0:3] if value in self.p1Hand]
            #print(len(self.p1Hand))
        cardStrings = ["None"] * 5

        #print(len(redhand))

        if len(redhand) > 0: cardStrings[0] = redhand[0].name + "(u: {}, r:{}, d:{}, l:{})".format(redhand[0].up, redhand[0].right, redhand[0].down, redhand[0].left)
        if len(redhand) > 1: cardStrings[1] = redhand[1].name + "(u: {}, r:{}, d:{}, l:{})".format(redhand[1].up, redhand[1].right, redhand[1].down, redhand[1].left)
        if len(redhand) > 2: cardStrings[2] = redhand[2].name + "(u: {}, r:{}, d:{}, l:{})".format(redhand[2].up, redhand[2].right, redhand[2].down, redhand[2].left)
        if len(redhand) > 3: cardStrings[3] = redhand[3].name + "(u: {}, r:{}, d:{}, l:{})".format(redhand[3].up, redhand[3].right, redhand[3].down, redhand[3].left)
        if len(redhand) > 4: cardStrings[4] = redhand[4].name + "(u: {}, r:{}, d:{}, l:{})".format(redhand[4].up, redhand[4].right, redhand[4].down, redhand[4].left)
        secretUnplayedCards = 0
        if (self.turn % 2 == 0):
            secretUnplayedCards = len([value for value in self.initialP2Hand[3:] if value in self.p2Hand])
        else:
            secretUnplayedCards = len([value for value in self.initialP1Hand[3:] if value in self.p1Hand])
        #print(secretUnplayedCards)
        for i in range(secretUnplayedCards):
            cardStrings[i + len(redhand)] = "Hidden"

        print(sb)
        print("You can see your opponent has: (1) {}, (2) {}, (3) {}, (4) {}, (5) {}".format(cardStrings[0], cardStrings[1], cardStrings[2], cardStrings[3], cardStrings[4]))

    def playCard(self, card, pos):
        self.gameboard[pos] = GameSquare(card, self.turn, pos)
        self.attemptFlip(card, pos)
        if (self.turn % 2 == 0):
            if card in self.p1Hand:
                self.p1Hand.remove(card)
        else:
            if card in self.p2Hand:
                self.p2Hand.remove(card)
        self.turn = self.turn + 1
        #print(self.turn)

    def attemptFlip(self, card, pos):
        #topface
        if (pos > 2):
            if (self.gameboard[pos - 3] is not None):
                if (self.gameboard[pos - 3].card.down < card.up):
                    self.gameboard[pos - 3].owner = self.turn % 2
                    if (self.turn % 2 == 0):
                        self.score+=1
                    else:
                        self.score-=1
        #bottomface
        if (pos < 6):
            if (self.gameboard[pos + 3] is not None):
                if (self.gameboard[pos + 3].card.up < card.down):
                    self.gameboard[pos + 3].owner = self.turn % 2
                    if (self.turn % 2 == 0):
                        self.score+=1
                    else:
                        self.score-=1
        #leftface
        if (pos % 3 > 0):
            if (self.gameboard[pos - 1] is not None):
                if (self.gameboard[pos - 1].card.right < card.left):
                    self.gameboard[pos - 1].owner = self.turn % 2
                    if (self.turn % 2 == 0):
                        self.score+=1
                    else:
                        self.score-=1

        #rightface
        if (pos % 3 < 2):
            if (self.gameboard[pos + 1] is not None):
                if (self.gameboard[pos + 1].card.left < card.right):
                    self.gameboard[pos + 1].owner = self.turn % 2
                    if (self.turn % 2 == 0):
                        self.score+=1
                    else:
                        self.score-=1


    def toFileString(self, allCards):
        fileString = "P1 Deck: "
        for i in range(len(self.initialP1Hand)):
            fileString += self.initialP1Hand[i].name
            if (i != 4):
                fileString += " | "
        fileString += "\nP2 Deck: "
        for i in range(len(self.initialP1Hand)):
            fileString += self.initialP2Hand[i].name
            if (i != 4):
                fileString += " | "
        fileString += "\nCards played: "
        for i in range(9):
            for j in range(9):
                if (self.gameboard[j] is not None):
                    if (i == self.gameboard[j].turnPlayed):
                        fileString += self.gameboard[j].card.name + ": " + str(self.gameboard[j].pos % 3) + ", " + str(self.gameboard[j].pos // 3)
                        fileString += " | "
        if (self.score > 0):
            fileString += "P1 Win"
        elif (self.score == 0):
            fileString += "Tie"
        else:
            fileString += "P1 Loss"
        return fileString
class GameSquare(object):
    def __init__(self, card, turn, pos):
        self.card = card
        self.turnPlayed = turn
        self.pos = pos
        self.owner = turn % 2

    def clone(self):
        return GameSquare(self.card, self.turnPlayed, self.pos)
