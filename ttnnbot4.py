#TODO: Fix the fucking naming conventions
#TODO: Get a better file management system
#TODO: maybe fix the MCTS's selection better?

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
import math

def get_bot_deck(deckID):
    normal, legendary, allcards, cardnamedict = make_card_data()
    with open("ttnn/decks/networkdeckchoices{}.txt".format(deckID)) as f:
        carddatarawunsplit = f.read()
    carddataraw = carddatarawunsplit.split("\n")
    line = carddataraw[random.randint(0, len(carddataraw) - 2)]
    cardNames = line[line.index(":") + 1:].replace(" ", "").replace(":", "").split("|")

    return [allcards[cardnamedict[cardNames[0]]], allcards[cardnamedict[cardNames[1]]], allcards[cardnamedict[cardNames[2]]], allcards[cardnamedict[cardNames[3]]], allcards[cardnamedict[cardNames[4]]]]
#helper methods to manage card files
def make_modified_cardlist(cutoff, statsfile, starterfile = 'cards.txt', outputfile = 'cardslimited.txt'):
    normal, legendary, allcards, cardnamedict = make_card_data(cardfile = starterfile)

    with open(statsfile) as f:
        carddatarawunsplit = f.read()
    carddataraw = carddatarawunsplit.split("\n")
    with open(outputfile, "w") as fo:
        for i in range(0, len(carddataraw) - 1):
            temp = carddataraw[i].split(":")[1]
            if float(temp[0:temp.index("%")]) > cutoff:
                card = allcards[cardnamedict[carddataraw[i].split(":")[0]]]
                fo.write("{},{},{},{},{},{}\n".format(card.name, card.stars, card.left, card.up, card.right, card.down))

def make_card_data(cardfile = 'cards.txt'):
    with open(cardfile) as f:
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

    def __init__(self, currentModelID, cardfile = 'cards.txt'):
        self.testnet = []
        self.currentModelID = currentModelID

        self.normal, self.legendary, self.allcards, self.cardnamedict = make_card_data(cardfile = cardfile)
        self.net = [] #Array if you want to make an array of networks (currently singleton array)
        self.fivestarnet = []
        self.playerrecord = [0,0,0] #Record of player wins and losses
        '''
        The network architecture is as follows:

        It accepts a 5 x 29 array of inputs representing the 29 cards on the board

        x is card data, y is which card

        x indexes 0 through 3 are up,left,down,right and 4 is owner

        y indexes 0 through 4 are p1 hand
        y indexes 5 through 9 are p2 hand
        y indexes 9 through 19 are the board
        y indexes 20 through 24 are p1 hand (again)
        y indexes 25 through 29 are p1 hand (again)

        We give the hand data again after the board data so that it can be in range for the convultion of the board with hand data

        We use 45 feature maps for each potentially desirable board state

        We use a layer with a high rate of dropout to prevent overfitting on small data batches

        We then flatten and let the last two fully connected layers figure it out

        TODO: Consider adding more convolution layers
        '''
        for i in range(1):


            model = keras.models.Sequential()

            model.add(keras.layers.Conv2D(45, (1, 9), activation='relu', input_shape=(5, 29, 1), name = "input"))
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



            model1 = keras.models.Sequential()

            model1.add(keras.layers.Conv2D(45, (1, 9), activation='relu', input_shape=(5, 29, 1), name = "input"))
            model1.add(keras.layers.BatchNormalization())

            #model.add(keras.layers.Conv2D(16, (1, 1), activation='relu', input_shape=(5, 20, 1), name = "input"))
            #model.add(keras.layers.BatchNormalization())
            model1.add(keras.layers.Dropout(.5))
            model1.add(keras.layers.Flatten())
            model1.add(keras.layers.Dense(45, activation='softmax'))
            model1.add(keras.layers.Dense(4, activation='softmax'))
            model1.add(keras.layers.Reshape((1, 4)))

            model1.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            model1.summary()
            self.fivestarnet.append(model1)
            #self.net[i].large_weight_initializer()
        '''
        Takes the gamestate and predicts the hidden 5 star card in your opponent's hand. Oddly accurate after comparing output * 29
        '''
        if (currentModelID == -1):
            self.currentModelID = 0
            tf.keras.models.save_model(self.net[0], "ttnn/finalnetwork4tf{}.txt".format(self.currentModelID))
            tf.keras.models.save_model(self.fivestarnet[0], "ttnn/finalnetwork4tffivestar{}.txt".format(self.currentModelID))
        self.load()
        for i in range(1):
            self.testnet.append(tf.keras.models.load_model("ttnn/finalnetwork4tf{}.txt".format(self.currentModelID)))
        random.seed(datetime.datetime.now()) #quality method of seeding random hand generation
        self.bestCardsInTests = [0] * len(self.allcards)
        self.totalCardsInTests = [0] * len(self.allcards)
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

    #method for convenience of hand generation
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


    #reads games from a file
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
            p1Deck = [self.allcards[self.cardnamedict[value]] for value in gamedata[i][len("P1 Deck: "):].replace(" ", "").replace("(Hidden)", "").split("|")]
            p2Deck = [self.allcards[self.cardnamedict[value]] for value in gamedata[i + 1][len("P2 Deck: "):].replace(" ", "").replace("(Hidden)", "").split("|")]
            gameboardData = gamedata[i + 2][len("Cards played: "): -1].replace(" ", "").replace("(Blue)", "").replace("(Red)", "").split("|")
            gameboard = Gameboard(p1Deck, p2Deck)
            for j in range(9):
                if (i + 3 < len(gamedata)):
                    if (("P2L" in gameboardData[-1] or ("P1W" in gameboardData[-1] or "T" in gameboardData[-1]))  and j % 2 == 1):
                        cardName, posstring = gameboardData[j].split(":")
                        pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])
                        gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)
                        continue
                    if ((("P2W" in gameboardData[-1]) or ("P1L" in gameboardData[-1] or "T" in gameboardData[-1])) and j % 2 == 0):
                        cardName, posstring = gameboardData[j].split(":")
                        pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])
                        gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)
                        continue
                cardName, posstring = gameboardData[j].split(":")
                pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])

                inputForNN = [0] * 145
                outputForNN = [0] * 9 * 5

                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, hideCards = False)
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
            npinput = np.reshape(np.array(nplistIn[j], order='F', ndmin = 4), [(len(nplistIn[j]) // 145), 5, 29, 1])
            npoutput = np.reshape(np.array(nplistOut[j], order='F', ndmin = 4), [(len(nplistOut[j]) // 45), 1, 45])
            nplist[0] = (npinput, npoutput)
        print(npinput.shape)
        return nplist
    '''
    Formats game data for the
    '''
    def formatGameFiveStar(self, gamedatafile):

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
            p1Hidden = [False] * 5
            p2Hidden = [False] * 5
            cardNamesHiddenP1 = gamedata[i][len("P1 Deck: "):].replace(" ", "").split("|")
            cardNamesHiddenP2 = gamedata[i + 1][len("P2 Deck: "):].replace(" ", "").split("|")
            for j in range(5):
                p1Hidden[j] = cardNamesHiddenP1[j].find("(Hidden)") >= 0
                p2Hidden[j] = cardNamesHiddenP2[j].find("(Hidden)") >= 0
            p1Deck = [self.allcards[self.cardnamedict[value]] for value in gamedata[i][len("P1 Deck: "):].replace(" ", "").replace("(Hidden)", "").split("|")]
            p2Deck = [self.allcards[self.cardnamedict[value]] for value in gamedata[i + 1][len("P2 Deck: "):].replace(" ", "").replace("(Hidden)", "").split("|")]
            gameboardData = gamedata[i + 2][len("Cards played: "): -1].replace(" ", "").replace("(Blue)", "").replace("(Red)", "").replace("(Hidden)", "").split("|")
            gameboard = Gameboard(p1Deck, p2Deck)
            gameboard.p1Hidden = p1Hidden
            gameboard.p2Hidden = p2Hidden
            for j in range(9):
                if (i + 3 < len(gamedata)):
                    if (("P2L" in gameboardData[-1] or ("P1W" in gameboardData[-1] or "T" in gameboardData[-1]))  and j % 2 == 1):
                        cardName, posstring = gameboardData[j].split(":")
                        pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])
                        gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)
                        continue
                    if ((("P2W" in gameboardData[-1]) or ("P1L" in gameboardData[-1] or "T" in gameboardData[-1])) and j % 2 == 0):
                        cardName, posstring = gameboardData[j].split(":")
                        pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])
                        gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)
                        continue
                cardName, posstring = gameboardData[j].split(":")
                pos = int(posstring.split(",")[0]) + 3 * int(posstring.split(",")[1])

                inputForNN = [0] * 145
                outputForNN = [0] * 4

                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict)
                if (j % 2 == 0):
                    card = None
                    for i in range(len(gameboard.p1Hand)):
                        card = gameboard.p1Hand[i]
                        if (card.stars == 5):
                            sum = (card.left / 10 + card.up / 10 + card.right / 10 + card.down / 10)
                            outputForNN[0] = card.left / 10 / sum
                            outputForNN[1] = card.up / 10 / sum
                            outputForNN[2] = card.right / 10 / sum
                            outputForNN[3] = card.down / 10 / sum
                            break
                    if outputForNN[0] == 0:
                        continue

                else:
                    card = None
                    for i in range(len(gameboard.p2Hand)):
                        card = gameboard.p2Hand[i]
                        if (card.stars == 5):
                            sum = (card.left / 10 + card.up / 10 + card.right / 10 + card.down / 10)
                            outputForNN[0] = card.left / 10 / sum
                            outputForNN[1] = card.up / 10 / sum
                            outputForNN[2] = card.right / 10 / sum
                            outputForNN[3] = card.down / 10 / sum
                            break
                    if outputForNN[0] == 0:
                        continue


                gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)

                if (nplistIn[0] == None):
                    nplistIn[0] = inputForNN
                    nplistOut[0] = outputForNN
                else:
                    nplistIn[0] += inputForNN
                    nplistOut[0] += outputForNN
        for j in range(1):
            npinput = np.reshape(np.array(nplistIn[j], order='F', ndmin = 4), [(len(nplistIn[j]) // 145), 5, 29, 1])
            npoutput = np.reshape(np.array(nplistOut[j], order='F', ndmin = 4), [(len(nplistOut[j]) // 4), 1, 4])
            nplist[0] = (npinput, npoutput)
        print(npinput.shape)
        return nplist

    def fiveStarWildToCard(self, wild):
        closestValue = 100000
        closestCard = None
        for i in range(len(self.legendary)):
            if (abs(wild.left - self.legendary[i].left) + abs(wild.up - self.legendary[i].up) + abs(wild.right - self.legendary[i].right) + abs(wild.down - self.legendary[i].down) < closestValue):
                closestValue = abs(wild.left - self.legendary[i].left) + abs(wild.up - self.legendary[i].up) + abs(wild.right - self.legendary[i].right) + abs(wild.down - self.legendary[i].down)
                closestCard = self.legendary[i]
        return closestCard

    def trainFiveStar(self, iterations = 5, gameNumber = 0):
        nplist = self.formatGameFiveStar("ttnn/thousandgames{}.txt".format(gameNumber))
        datasetIn, datasetOut = nplist[0]
        self.fivestarnet[0].fit(datasetIn, datasetOut, epochs=iterations)
        datasetOut = np.reshape(datasetOut, -1).tolist()
        output = np.reshape(self.fivestarnet[0].predict(np.reshape(np.array(datasetIn, order='F', ndmin = 4), (-1, 5, 29, 1))), -1).tolist()
        for i in range(0, np.size(output) // 4 - 1, 1):
            card = self.fiveStarWildToCard(Card(["", 5, output[i * 4 + 0] * 29, output[i * 4 + 1] * 29, output[i * 4 + 2] * 29, output[i * 4 + 3] * 29]))
            supposedCard = self.fiveStarWildToCard(Card(["", 5, datasetOut[i * 4 + 0] * 29, datasetOut[i * 4 + 1] * 29, datasetOut[i * 4 + 2] * 29, datasetOut[i * 4 + 3] * 29]))
            print("Card predicted is: {}. Card was probably: {}".format(card.name, supposedCard.name))
        #print(datasetOut)
        #Ftf.keras.models.save_model(self.fivestarnet[0], "ttnn/finalnetwork4tffivestar{}.txt".format(self.currentModelID))
        tf.keras.models.save_model(self.fivestarnet[0], "ttnn/finalnetwork4tffivestar{}.txt".format(self.currentModelID + 1))
    '''
    Call this to train the network
    Optional parameters are depretacted
    '''
    def mainmethod(self, iterations = 5, learningrate = 0.2, numgames=100, shouldTrainFiveStar = False):

        for i in range(1000):
            handdata = self.play_games(number_of_games = numgames, outputfile = "ttnn/thousandgames{}.txt".format(i))
            nplist = self.formatGame("ttnn/thousandgames{}.txt".format(i))

            for j in range(1):
                print("Processing network {}".format(j))
                datasetIn, datasetOut = nplist[0]
                self.net[0].fit(datasetIn, datasetOut, epochs=iterations)
                if shouldTrainFiveStar: self.trainFiveStar(iterations = iterations, gameNumber = i)
                tf.keras.models.save_model(self.net[0], "ttnn/finalnetwork4tf{}.txt".format(self.currentModelID + 1))
                print("Saved network {} to file!".format(0))
            self.test_network(handdata, number_of_games = numgames, shouldUpdate = True)


    '''
    Modified version of play_games(...), meant for a human player to face the bot
    Callable from a shell, directory is deprecated, plays a single game, records score
    '''
    def playAgainstBot(self, directory = "nn10/", playerHand = None):

        stringedGamesPlayed = ""
        #testnet = []
        #for i in range(9):
            #testnet.append(network2.load("ttnn/archive/" + directory + "finalnetwork{}.txt".format(i)))
        p1 = self.generate_random_hand()
        p2 = self.generate_random_hand()
        #describes if the HUMAN PLAYER is P1
        youAreP1 = isP1 = random.randint(0,1)==0
        try:
            if playerHand is not None:

                if youAreP1:
                #p2 = [self.allcards[self.cardnamedict["Ysayle"]], self.allcards[self.cardnamedict["Moglin"]], self.allcards[self.cardnamedict["TheGriffin"]], self.allcards[self.cardnamedict["Lightning"]], self.allcards[self.cardnamedict["Arenvald"]]]
                #p2 = [self.allcards[self.cardnamedict["CecilHarvey"]], self.allcards[self.cardnamedict["Asahi"]], self.allcards[self.cardnamedict["Louhi"]], self.allcards[self.cardnamedict["Vedrfolnir"]], self.allcards[self.cardnamedict["Coeurlregina"]]]
                #p2 = get_bot_deck(self.currentModelID)
                    p1 = [self.allcards[self.cardnamedict[playerHand[0]]], self.allcards[self.cardnamedict[playerHand[1]]], self.allcards[self.cardnamedict[playerHand[2]]], self.allcards[self.cardnamedict[playerHand[3]]], self.allcards[self.cardnamedict[playerHand[4]]]]


                else:
                #p1 = [self.allcards[self.cardnamedict["Ysayle"]], self.allcards[self.cardnamedict["Moglin"]], self.allcards[self.cardnamedict["TheGriffin"]], self.allcards[self.cardnamedict["Lightning"]], self.allcards[self.cardnamedict["Arenvald"]]]
                #p1 = [self.allcards[self.cardnamedict["CecilHarvey"]], self.allcards[self.cardnamedict["Asahi"]], self.allcards[self.cardnamedict["Louhi"]], self.allcards[self.cardnamedict["Vedrfolnir"]], self.allcards[self.cardnamedict["Coeurlregina"]]]
                #p1 = get_bot_deck(self.currentModelID)
                    p2 = [self.allcards[self.cardnamedict[playerHand[0]]], self.allcards[self.cardnamedict[playerHand[1]]], self.allcards[self.cardnamedict[playerHand[2]]], self.allcards[self.cardnamedict[playerHand[3]]], self.allcards[self.cardnamedict[playerHand[4]]]]
        except:
            print("Error parsing custom deck!")
            return
        #p1 = get_bot_deck(self.currentModelID)
        #p2 = get_bot_deck(self.currentModelID)
        random.shuffle(p1)
        random.shuffle(p2)
        gameboard = Gameboard(p1, p2)
        for j in range(9):
            #inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = not youAreP1)
            #outputFromNN = None
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
                    print("Thinking...")
                    #outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                    cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.net[0], timeToEstimate = 7000)
                    #print("Odds of bot victory: {}%".format(int(100 * mctsResults[0] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    #print("Odds of bot tie: {}%".format(int(100 * mctsResults[1] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    #print("Odds of bot loss: {}%".format(int(100 * mctsResults[2] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    print(cardToPlay.name)
                    gameboard.playCard(cardToPlay, posToPlay)

            else:
                if youAreP1:
                    print("Thinking...")
                    #outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                    cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.net[0], timeToEstimate = 7000)
                    #print("Odds of bot victory: {}%".format(int(100 * mctsResults[0] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    #print("Odds of bot tie: {}%".format(int(100 * mctsResults[1] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    #print("Odds of bot loss: {}%".format(int(100 * mctsResults[2] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    print(cardToPlay.name)
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
        if youAreP1:
            if (gameboard.score > 0):
                print("You win!")
                self.playerrecord[0] += 1
            if (gameboard.score == 0):
                print("Tie!")
                self.playerrecord[1] += 1
            if (gameboard.score < 0):
                print("You lose!")
                self.playerrecord[2] += 1
        else:
            if (gameboard.score < 0):
                print("You win!")
                self.playerrecord[0] += 1
            if (gameboard.score == 0):
                print("Tie!")
                self.playerrecord[1] += 1
            if (gameboard.score > 0):
                print("You lose!")
                self.playerrecord[2] += 1
        print("Your record is: {}W/{}T/{}L.".format(self.playerrecord[0], self.playerrecord[1], self.playerrecord[2]))


    def MCTS2Top(self, gameboard, canSeeWholeHand, networkToPredictWith, timeToEstimate = 2000, training = False, initialNodes = 6):

        startTime = int(round(time.time() * 1000))
        #we (the predictor) are p1 if we can see our entire hand on an even turn or if we cannot see our entire hand on a odd (p2) turn
        isP1 = (canSeeWholeHand and gameboard.turn % 2 == 0) or ((not canSeeWholeHand) and gameboard.turn % 2 == 1)
        inputFor5SNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
        fiveStarOutputFromNN = np.reshape(self.fivestarnet[0].predict(np.reshape(np.array(inputFor5SNN, order='F', ndmin = 4), (-1, 5, 29, 1))), 4).tolist()
        fiveStarPrediction = self.fiveStarWildToCard(Card(["Five Star Prediction", 5, fiveStarOutputFromNN[0] * 29, fiveStarOutputFromNN[1] * 29, fiveStarOutputFromNN[2] * 29, fiveStarOutputFromNN[3] * 29]))

        inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, fiveStarPrediction = fiveStarPrediction)
        outputFromNN = networkToPredictWith.predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
        topNodes = []
        b = np.reshape(outputFromNN, (45)).tolist()
         #when training, slightly randomize the outputs
        if training:
             for j in range(45): b[j] += (random.random() * 0.1 - 0.05)

        indexesInOrder = {}
        sumScores = [0, 0, 0] #list of summed scores to send to the parent MCTS call
        validIndexes = []
        i1 = 0
        #we now sort the list of indexes, but create another list that tells us their original indexing
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


        while (len(validIndexes) < initialNodes and i1 < 45):

            topChoiceIndex = indexesInOrder[i1]
            card = None

            if (gameboard.turn % 2 == 0):
                card = gameboard.initialP1Hand[topChoiceIndex % 5]
            else:
                card = gameboard.initialP2Hand[topChoiceIndex % 5]


            pos = topChoiceIndex // 5
            if (gameboard.turn % 2 == 0):
                if not card in gameboard.p1Hand:
                    i1+=1
                    continue
            else:
                if not card in gameboard.p2Hand:
                    i1+=1
                    continue
            if (not gameboard.isValidMove(card, pos)):
                i1+=1
                continue


            validIndexes.append(topChoiceIndex)
            i1+=1

                #create the gameboards describing the state of the game on the next potential turns
        gbCopies = []

        for i in range(len(validIndexes)):
            redhand = [None]
            card = None

            if (gameboard.turn % 2 == 0):
                card = gameboard.initialP1Hand[validIndexes[i] % 5]
            else:
                card = gameboard.initialP2Hand[validIndexes[i] % 5]
            pos = validIndexes[i] // 5

            gbCopy = gameboard.clone()
            if (canSeeWholeHand):
                if gameboard.turn % 2 == 0:
                    #in this case we are calculating for the hand without hidden cards
                    redhand = [value for value in gameboard.initialP1Hand]
                    card = redhand[validIndexes[i] % 5]
                else:
                    #in this case we are predicting for the hand with hidden cards
                    redhand = [value for value in gameboard.initialP2Hand]
                    card = redhand[validIndexes[i] % 5]
            else:
                if gameboard.turn % 2 == 0:
                    redhand = [value for value in gameboard.initialP1Hand]
                    #if we cannot see the whole hand, and its turn 0, 2, 4, 6, or 8, then we are p2
                    redhand = gameboard.flipHiddenCards(redhand, False, self.allcards, self.normal, self.legendary, self.cardnamedict, fiveStarPrediction = fiveStarPrediction)
                    card = redhand[validIndexes[i] % 5]
                    gbCopy.p1Hand.remove(gameboard.initialP1Hand[validIndexes[i] % 5])
                else:
                    redhand = [value for value in gameboard.initialP2Hand]
                    #if we cannot see the whole hand and its turn 1, 3, 5, or 7, then we are p1
                    redhand = gameboard.flipHiddenCards(redhand, True, self.allcards, self.normal, self.legendary, self.cardnamedict, fiveStarPrediction = fiveStarPrediction)
                    card = redhand[validIndexes[i] % 5]
                    gbCopy.p2Hand.remove(gameboard.initialP2Hand[validIndexes[i] % 5])


            gbCopy.playCard(card, pos)
            gbCopies.append(gbCopy)

        for i in range(len(validIndexes)):
            data, card, pos, history = self.MCTS2(gbCopies[i], 9, canSeeWholeHand, networkToPredictWith, training = training, fiveStarPrediction = fiveStarPrediction, desiredValidIndexes = 1, tree = None)
            history = [validIndexes[i]] + history
            if (len(history) != 0):
                topNodes.append(TreeNode(data, history, isP1, gbCopies[i].turn))
            else:
                for i in range(9):
                    if gbCopies[0].gameboard[i].turnPlayed == 8:
                        return (gbCopies[0].gameboard[i].card, gbCopies[0].gameboard[i].pos)

        searchExhausted = False
        while int(round(time.time() * 1000)) - startTime < timeToEstimate:
            gbCopy = gameboard.clone()
            '''
            Evaluate Top Node Values
            '''
            bestNode = -100000
            bestNodeValue = -100000
            totalNodes = 0
            for i in range(len(topNodes)):
                totalNodes += topNodes[i].total

            for i in range(len(topNodes)):
                nodeValue = topNodes[i].wins / topNodes[i].total + (math.log(totalNodes) / topNodes[i].total * 3 ) ** 0.5
                if nodeValue > bestNodeValue:
                    if not topNodes[i].isDead():
                        bestNode = i
                        bestNodeValue = nodeValue
            if (bestNode < 0): #this means we've exhausted our search
                searchExhausted = True #if this happens we need a new metric to decide the best node
                break
            #we use "not canSeeWholeHand" because we're talking about the turn after this one
            data, card, pos, history = self.MCTS2(gbCopies[bestNode], 9, not canSeeWholeHand, networkToPredictWith, training = training, fiveStarPrediction = fiveStarPrediction, desiredValidIndexes = 1, tree = topNodes[bestNode])
            topNodes[bestNode].addChild(data, history)
        bestNode = -100000
        bestNodeValue = -100000
        for i in range(len(topNodes)):
            nodeValue = topNodes[i].total
            if searchExhausted:
                nodeValue = topNodes[i].wins / topNodes[i].total
            if nodeValue > bestNodeValue:
                bestNode = i
                bestNodeValue = nodeValue
        return self.outputValueToCardAndPos(gameboard, topNodes[bestNode].choice)


    def MCTS2(self, gameboard, turnsToCheck, canSeeWholeHand, networkToPredictWith, training = False, shouldBeRandom = False, topLayer = True, layerOutput = None, fiveStarPrediction = None, desiredValidIndexes = 1, playToTie = False, tree = None):
        startTime = int(round(time.time() * 1000))
        isP1 = (canSeeWholeHand and gameboard.turn % 2 == 0) or ((not canSeeWholeHand) and gameboard.turn % 2 == 1)
        #special conditions? Ie do we need to continue searching
        if(turnsToCheck <= 0 or gameboard.turn >= 8):
            #if canSeeWholeHand on turn 9, then we are p1
            #if not canSeeWholeHand on turn 9, then we are p2
            if isP1:
                #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))
                if gameboard.score > 0: return ([1, 0, 0], None, -1, [])
                elif gameboard.score == 0: return ([0, 1, 0], None, -1, [])
                else: return ([0, 0, 1], None,-1, [])
            else:
                #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))
                if gameboard.score < 0: return ([1, 0, 0], None,-1, [])
                elif gameboard.score == 0: return ([0, 1, 0], None,-1, [])
                else: return ([0, 0, 1], None,-1, [])



        #outputFromNN is the current boardstate fed to the neural network
        outputFromNN = None
        if layerOutput is not None:
            outputFromNN = layerOutput
        else:
            inputFor5SNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
            fiveStarOutputFromNN = np.reshape(self.fivestarnet[0].predict(np.reshape(np.array(inputFor5SNN, order='F', ndmin = 4), (-1, 5, 29, 1))), 4).tolist()
            inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, fiveStarPrediction = self.fiveStarWildToCard(Card(["Five Star Prediction", 5, fiveStarOutputFromNN[0] * 29, fiveStarOutputFromNN[1] * 29, fiveStarOutputFromNN[2] * 29, fiveStarOutputFromNN[3] * 29])))
            outputFromNN = networkToPredictWith.predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
            #print("5 star is believed to be: {}".format(self.fiveStarWildToCard(Card(["Five Star Prediction", 5, fiveStarOutputFromNN[0] * 29, fiveStarOutputFromNN[1] * 29, fiveStarOutputFromNN[2] * 29, fiveStarOutputFromNN[3] * 29])).name))



        scores = [[0, 0, 0]] * desiredValidIndexes #list of subscores


        #reshaped list of the output from the NN
        b = np.reshape(outputFromNN, (45)).tolist()
         #when training, slightly randomize the outputs
        if training:
             for i in range(45): b[i] += (random.random() * 0.1 - 0.05)

        indexesInOrder = {}
        sumScores = [0, 0, 0] #list of summed scores to send to the parent MCTS call

        #we now sort the list of indexes, but create another list that tells us their original indexing
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

        childCardChoies = []

        i1 = 0 #index for outputs for our while loop
        validIndexes = [] #the list of choices by the bot that can actually be played

        #now we go through all 45 possible outputs, check which are possible to be played, and add up to the best 6 with priority to who is checked given by the NN output
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

            if tree is not None:
                if tree.getChild(topChoiceIndex) is not None:
                    if (tree.getChild(topChoiceIndex).isDead()):
                        i1+=1
                        continue



            validIndexes.append(topChoiceIndex)
            i1+=1

        #create the gameboards describing the state of the game on the next potential turns
        gbCopies = []

        for i in range(len(validIndexes)):
            redhand = [None]
            card = None

            if (gameboard.turn % 2 == 0):
                card = gameboard.initialP1Hand[validIndexes[i] % 5]
            else:
                card = gameboard.initialP2Hand[validIndexes[i] % 5]
            pos = validIndexes[i] // 5
            gbCopy = gameboard.clone()
            if (canSeeWholeHand):
                if gameboard.turn % 2 == 0:
                    redhand = [value for value in gameboard.initialP1Hand]
                    card = redhand[validIndexes[i] % 5]
                else:
                    redhand = [value for value in gameboard.initialP2Hand]
                    card = redhand[validIndexes[i] % 5]
            else:
                if gameboard.turn % 2 == 0:
                    redhand = [value for value in gameboard.initialP1Hand]
                    #print(len(bluehand))
                    #if we cannot see the whole hand, and its turn 0, 2, 4, 6, or 8, then we (the predictor) are p2
                    redhand = gameboard.flipHiddenCards(redhand, False, self.allcards, self.normal, self.legendary, self.cardnamedict, fiveStarPrediction = fiveStarPrediction)
                    card = redhand[validIndexes[i] % 5]
                    #its important that the hidden card actually gets played from the original hand, else it'll be available next time!
                    gbCopy.p1Hand.remove(gameboard.initialP1Hand[validIndexes[i] % 5])
                else:
                    redhand = [value for value in gameboard.initialP2Hand]
                    #conversely if we cannot see the whole hand and its an odd turn, we (the predictor) must be p1
                    redhand = gameboard.flipHiddenCards(redhand, True, self.allcards, self.normal, self.legendary, self.cardnamedict, fiveStarPrediction = fiveStarPrediction)
                    card = redhand[validIndexes[i] % 5]
                    #its important that the hidden card actually gets played from the original hand, else it'll be available next time!
                    gbCopy.p2Hand.remove(gameboard.initialP2Hand[validIndexes[i] % 5])
            gbCopy.playCard(card, pos)

            gbCopies.append(gbCopy)

        history = [validIndexes[0]]
        if turnsToCheck > 1:
            #list of all inputs for the neural network
            inputsForNN = []
            #we now create the new inputs for the next depth of the MCTS we do it this way because it saves computation time in Tensorflow
            fiveStarOutputsFromNN = []
            inputsFor5SNN = []
            for i in range(len(validIndexes)):
                inputsFor5SNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
            fiveStarOutputsFromNN = np.reshape(self.fivestarnet[0].predict(np.reshape(np.array(inputsFor5SNN, order='F', ndmin = 4), (-1, 5, 29, 1))), (len(validIndexes) * 4)).tolist()
            for i in range(len(validIndexes)):
                #inputsFor5SNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
                inputsForNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, fiveStarPrediction = self.fiveStarWildToCard(Card(["Five Star Prediction", 5, 29 * fiveStarOutputsFromNN[i * 4], 29 * fiveStarOutputsFromNN[i * 4 + 1], 29 * fiveStarOutputsFromNN[i * 4 + 2], 29 * fiveStarOutputsFromNN[i * 4 + 3]])))

            #feed data to our network, we reshape to make sure its the right format for both the input and the output
            outputsFromNN = np.reshape(networkToPredictWith.predict(np.reshape(np.array(inputsForNN, order='F', ndmin = 4), (-1, 5, 29, 1))), (len(validIndexes), 45))

            #now we ship off the values to next layer of the MCTS
            for i in range(len(validIndexes)):
                fiveStarPred = self.fiveStarWildToCard(Card(["Five Star Prediction", 5, 29 * fiveStarOutputsFromNN[i * 4], 29 * fiveStarOutputsFromNN[i * 4 + 1], 29 * fiveStarOutputsFromNN[i * 4 + 2], 29 * fiveStarOutputsFromNN[i * 4 + 3]]))
                if tree is not None:

                    mctsdata, card, pos, childHistory = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, layerOutput = outputsFromNN[i], fiveStarPrediction = fiveStarPred, desiredValidIndexes = desiredValidIndexes, tree = tree.getChild(validIndexes[i]))
                    history += childHistory
                    scores[i] = mctsdata
                else:
                    mctsdata, card, pos, childHistory = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, layerOutput = outputsFromNN[i], fiveStarPrediction = fiveStarPred, desiredValidIndexes = desiredValidIndexes, tree = None)
                    history += childHistory
                    scores[i] = mctsdata


        else:
            #then next turn will not be using its output data from the neural network, so we simply have it skip calculating the final turn
            for i in range(len(validIndexes)):
                if tree is not None:
                    mctsdata, card, pos, childHistory  = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, tree = tree.getChild(validIndexes[i]))
                    history += childHistory
                    scores[i] = mctsdata
                else:
                    mctsdata, card, pos, childHistory  = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, tree = None)
                    history += childHistory
                    scores[i] = mctsdata

        #score summation / choice evaluation
        bestScore = -10000000
        bestIndex = -1
        scoreIndex = -70
        #mostly useless atm
        for i in range(len(validIndexes)):
            indexScore = 0
            if canSeeWholeHand:
                if gameboard.turn % 2 == 0:
                    indexScore = scores[i][0] - scores[i][2] - 0.99 * scores[i][1]
                else:
                    indexScore = -(scores[i][0] - scores[i][2] - 0.99 * scores[i][1])
            else:
                if gameboard.turn % 2 == 1:
                    indexScore = scores[i][0] - scores[i][2] - 0.99 * scores[i][1]
                else:
                    indexScore = -(scores[i][0] - scores[i][2] - 0.99 * scores[i][1])
            if playToTie: indexScore = scores[i][0] - scores[i][2] + 1 * scores[i][1]
            if (bestScore < indexScore):
                bestIndex = validIndexes[i]
                bestScore = indexScore
                scoreIndex = i


        for i in range(len(validIndexes)):
            sumScores[0] += scores[i][0]
            sumScores[1] += scores[i][1]
            sumScores[2] += scores[i][2]

        card = None

        if (gameboard.turn % 2 == 0):
            card = gameboard.initialP1Hand[bestIndex % 5]
        else:
            card = gameboard.initialP2Hand[bestIndex % 5]

        pos = bestIndex // 5

        if (bestIndex == -1): #if for some reason we found no moves, just find a valid move! This should never happen, but if it does this will always generate a valid move.
            print("Failed to find a valid move! This is a bug!")
            for i in range(45):
                if (gameboard.turn % 2 == 0):
                    card = gameboard.initialP1Hand[i % 5]
                else:
                    card = gameboard.initialP2Hand[i % 5]


                pos = i // 5

                if (gameboard.isValidMove(card, pos)):
                    bestIndex = i
                    scoreIndex = 0
                    break



        #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))

        #if we're on the top layer, report scores with=
        return (sumScores, card, pos, history)

    '''
    MCTS: Monte Carlo Tree Search

    Implementation is taken from Alpha Go Zero. Ie we have a search depth and do not play games to completion

    returns: best choice of the network after MCTS refinement

    Essentially our implementation (with optimization) functions as such:

    (1) Initial Call to  MCTS --> (2) get an initial output neural network to work with --> (3) sort options -->
    (4) validate options --> (5) generate next layer --> (6) call next layer of MCTS --> goto (3) until you reach bottom --> add scores to find best move
    '''



    def MCTS(self, gameboard, turnsToCheck, isP1, training = False, shouldBeRandom = False, topLayer = True, layerOutput = None, fiveStarPrediction = None, desiredValidIndexes = 6, playToTie = False):
        startTime = int(round(time.time() * 1000))

        #special conditions? Ie do we need to continue searching
        if(turnsToCheck <= 0 or gameboard.turn >= 9):
            if isP1:
                #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))
                if gameboard.score > 0: return ([1, 0, 0], None, -1)
                elif gameboard.score == 0: return ([0, 1, 0], None, -1)
                else: return ([0, 0, abs(gameboard.score)], None,-1)
            else:
                #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))
                if gameboard.score < 0: return ([1, 0, 0], None,-1)
                elif gameboard.score == 0: return ([0, 1, 0], None,-1)
                else: return ([0, 0, abs(gameboard.score)], None,-1)



        #outputFromNN is the current boardstate fed to the neural network
        outputFromNN = None
        if layerOutput is not None:
            outputFromNN = layerOutput
        else:
            inputFor5SNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
            fiveStarOutputFromNN = np.reshape(self.fivestarnet[0].predict(np.reshape(np.array(inputFor5SNN, order='F', ndmin = 4), (-1, 5, 29, 1))), 4).tolist()
            inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, fiveStarPrediction = self.fiveStarWildToCard(Card(["Five Star Prediction", 5, fiveStarOutputFromNN[0] * 29, fiveStarOutputFromNN[1] * 29, fiveStarOutputFromNN[2] * 29, fiveStarOutputFromNN[3] * 29])))
            outputFromNN = networkToPredictWith.predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
            #print("5 star is believed to be: {}".format(self.fiveStarWildToCard(Card(["Five Star Prediction", 5, fiveStarOutputFromNN[0] * 29, fiveStarOutputFromNN[1] * 29, fiveStarOutputFromNN[2] * 29, fiveStarOutputFromNN[3] * 29])).name))



        scores = [[0, 0, 0]] * desiredValidIndexes #list of subscores


        #reshaped list of the output from the NN
        b = np.reshape(outputFromNN, (45)).tolist()
         #when training, slightly randomize the outputs
        if training:
             for i in range(45): b[i] += (random.random() * 0.1 - 0.05)

        indexesInOrder = {}
        sumScores = [0, 0, 0] #list of summed scores to send to the parent MCTS call

        #we now sort the list of indexes, but create another list that tells us their original indexing
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



        i1 = 0 #index for outputs for our while loop
        validIndexes = [] #the list of choices by the bot that can actually be played

        #now we go through all 45 possible outputs, check which are possible to be played, and add up to the best 6 with priority to who is checked given by the NN output
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

        #create the gameboards describing the state of the game on the next potential turns
        gbCopies = []

        for i in range(len(validIndexes)):
            redhand = [None]
            card = None

            if (gameboard.turn % 2 == 0):
                card = gameboard.initialP1Hand[validIndexes[i] % 5]
            else:
                card = gameboard.initialP2Hand[validIndexes[i] % 5]
            pos = validIndexes[i] // 5
            gbCopy = gameboard.clone()
            if (isP1):
                if gameboard.turn % 2 == 0:
                    redhand = [value for value in gameboard.initialP1Hand]
                    card = redhand[validIndexes[i] % 5]
                else:
                    redhand = [value for value in gameboard.initialP2Hand]
                    #print(len(redhand))
                    #print(len(redhand))
                    redhand = gameboard.flipHiddenCards(redhand, isP1, self.allcards, self.normal, self.legendary, self.cardnamedict, fiveStarPrediction = fiveStarPrediction)
                    card = redhand[validIndexes[i] % 5]
            else:
                if gameboard.turn % 2 == 0:
                    redhand = [value for value in gameboard.initialP1Hand]
                    #print(len(bluehand))
                    redhand = gameboard.flipHiddenCards(redhand, isP1, self.allcards, self.normal, self.legendary, self.cardnamedict, fiveStarPrediction = fiveStarPrediction)
                    card = redhand[validIndexes[i] % 5]
                else:
                    redhand = [value for value in gameboard.initialP2Hand]
                    card = redhand[validIndexes[i] % 5]

            gbCopy.playCard(card, pos)

            gbCopies.append(gbCopy)
        if turnsToCheck > 1:
            #list of all inputs for the neural network
            inputsForNN = []
            #we now create the new inputs for the next depth of the MCTS we do it this way because it saves computation time in Tensorflow
            fiveStarOutputsFromNN = []
            inputsFor5SNN = []
            for i in range(len(validIndexes)):
                inputsFor5SNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
            fiveStarOutputsFromNN = np.reshape(self.fivestarnet[0].predict(np.reshape(np.array(inputsFor5SNN, order='F', ndmin = 4), (-1, 5, 29, 1))), (len(validIndexes) * 4)).tolist()
            for i in range(len(validIndexes)):
                #inputsFor5SNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
                inputsForNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, fiveStarPrediction = self.fiveStarWildToCard(Card(["Five Star Prediction", 5, 29 * fiveStarOutputsFromNN[i * 4], 29 * fiveStarOutputsFromNN[i * 4 + 1], 29 * fiveStarOutputsFromNN[i * 4 + 2], 29 * fiveStarOutputsFromNN[i * 4 + 3]])))

            #feed data to our network, we reshape to make sure its the right format for both the input and the output
            outputsFromNN = np.reshape(self.net[0].predict(np.reshape(np.array(inputsForNN, order='F', ndmin = 4), (-1, 5, 29, 1))), (len(validIndexes), 45))

            #now we ship off the values to next layer of the MCTS
            for i in range(len(validIndexes)):
                mctsdata, card, pos = self.MCTS(gbCopies[i], turnsToCheck - 1, isP1, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, layerOutput = outputsFromNN[i], fiveStarPrediction = self.fiveStarWildToCard(Card(["Five Star Prediction", 5, 29 * fiveStarOutputsFromNN[i * 4], 29 * fiveStarOutputsFromNN[i * 4 + 1], 29 * fiveStarOutputsFromNN[i * 4 + 2], 29 * fiveStarOutputsFromNN[i * 4 + 3]])), desiredValidIndexes = desiredValidIndexes)
                scores[i] = mctsdata
        else:
            #then next turn will not be using its output data from the neural network, so we simply have it skip calculating the final turn
            for i in range(len(validIndexes)):
                mctsdata, card, pos = self.MCTS(gbCopies[i], turnsToCheck - 1, isP1, training = training, shouldBeRandom = shouldBeRandom, topLayer = False)
                scores[i] = mctsdata

        #score summation / choice evaluation
        bestScore = -10000000
        bestIndex = -1
        scoreIndex = -70
        for i in range(len(validIndexes)):
            indexScore = 0
            if isP1:
                if gameboard.turn % 2 == 0:
                    indexScore = scores[i][0] - scores[i][2] - 0.99 * scores[i][1]
                else:
                    indexScore = -(scores[i][0] - scores[i][2] - 0.99 * scores[i][1])
            else:
                if gameboard.turn % 2 == 1:
                    indexScore = scores[i][0] - scores[i][2] - 0.99 * scores[i][1]
                else:
                    indexScore = -(scores[i][0] - scores[i][2] - 0.99 * scores[i][1])
            if playToTie: indexScore = scores[i][0] - scores[i][2] + 1 * scores[i][1]
            if (bestScore < indexScore):
                bestIndex = validIndexes[i]
                bestScore = indexScore
                scoreIndex = i


        for i in range(len(validIndexes)):
            sumScores[0] += scores[i][0]
            sumScores[1] += scores[i][1]
            sumScores[2] += scores[i][2]

        card = None

        if (gameboard.turn % 2 == 0):
            card = gameboard.initialP1Hand[bestIndex % 5]
        else:
            card = gameboard.initialP2Hand[bestIndex % 5]

        pos = bestIndex // 5

        if (bestIndex == -1): #if for some reason we found no moves, just find a valid move! This should never happen, but if it does this will always generate a valid move.
            print("Failed to find a valid move! This is a bug!")
            for i in range(45):
                if (gameboard.turn % 2 == 0):
                    card = gameboard.initialP1Hand[i % 5]
                else:
                    card = gameboard.initialP2Hand[i % 5]


                pos = i // 5

                if (gameboard.isValidMove(card, pos)):
                    bestIndex = i
                    scoreIndex = 0
                    break



        #print ("Time to complete MCTS with searchlevel {}: {} ms".format(turnsToCheck, int(round(time.time() * 1000)) - startTime))

        #if we're on the top layer, report scores with
        if (layerOutput is None): return (scores[scoreIndex], card, pos)
        return (sumScores, card, pos)

    '''
    Method to test the network. Tests differently depending if shouldUpdate is True. Probably should be split into two methods.
    '''
    def test_network(self, handdata, directory = "nn1/", number_of_games = 1000, shouldUpdate = False):
        stringedGamesPlayed = ""

        winsAsP1 = 0
        winsAsP2 = 0
        tiesAsP1 = 0
        tiesAsP2 = 0
        lossesAsP1 = 0
        lossesAsP2 = 0
        outputString = ""
        numdeckstoTest = 10
        if shouldUpdate: numdeckstoTest = 1
        for i in range(number_of_games):
            if not shouldUpdate: print("Starting deck {} processing...".format(i))
            isP1 = random.randint(0,1)==0
            p1 = self.generate_random_hand()
            p2 = self.generate_random_hand()
            wins = 0
            for i1 in range(numdeckstoTest):
                if not shouldUpdate:
                    if isP1:
                        p2 = p1
                    else:
                        p1 = p2
                    isP1 = not isP1

                if isP1:
                    if not shouldUpdate:
                        #p1 = get_bot_deck(self.currentModelID)
                        p2 = [self.allcards[self.cardnamedict["Ysayle"]], self.allcards[self.cardnamedict["Cloud"]], self.allcards[self.cardnamedict["Hilda"]], self.allcards[self.cardnamedict["Estinien"]], self.allcards[self.cardnamedict["Asahi"]]]
                elif not shouldUpdate:
                    #p2 = get_bot_deck(self.currentModelID)
                        p1 = [self.allcards[self.cardnamedict["Ysayle"]], self.allcards[self.cardnamedict["Cloud"]], self.allcards[self.cardnamedict["Hilda"]], self.allcards[self.cardnamedict["Estinien"]], self.allcards[self.cardnamedict["Asahi"]]]
                random.shuffle(p1)
                random.shuffle(p2)
                gameboard = Gameboard(p1.copy(), p2.copy())
                for j in range(9):
                    inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict)
                    outputFromNN = None
                    cardToPlay, posToPlay = (None, 0)

                    if j % 2 == 0:
                        if isP1:
                            #outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.net[0])
                        else:
                            #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testnet[0])

                    else:
                        if isP1:
                            #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testnet[0])

                        else:
                            #outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.net[0])

                    #print(outputFromNN.shape)
                    gameboard.playCard(cardToPlay, posToPlay)


                if isP1:


                    if gameboard.score > 0:
                        wins += 1
                        if (wins > 4):
                            break
                        winsAsP1 +=1

                    elif gameboard.score == 0:
                        tiesAsP1 +=1
                    else:
                        lossesAsP1 += 1
                else:
                    if (gameboard.score < 0):
                        wins += 1
                        if (wins > 4):
                            break
                        winsAsP2 +=1
                    elif gameboard.score == 0:
                        tiesAsP2 +=1
                    else:
                        lossesAsP2 += 1
                if shouldUpdate:
                    print(gameboard.toFileString(self.allcards, isP1 = isP1))
            if (wins > 4):
                fileString = ""
                if isP1:
                    fileString += "P1 Deck: "
                    for i in range(len(gameboard.initialP1Hand)):
                        fileString += gameboard.initialP1Hand[i].name
                        if (i != 4):
                            fileString += " | "
                else:
                    fileString = "P1 Deck: "
                    for i in range(len(gameboard.initialP2Hand)):
                        fileString += gameboard.initialP2Hand[i].name
                        if (i != 4):
                            fileString += " | "
                outputString += fileString + "\n"
                print(fileString)



        if (winsAsP2 + winsAsP1) / (winsAsP2 + winsAsP1 + lossesAsP1 + lossesAsP2) > 0.54 and shouldUpdate:
            self.currentModelID += 1
            del self.testnet[0]
            self.testnet = [None]
            for i in range(1):
                self.testnet[0] = tf.keras.models.load_model("ttnn/finalnetwork4tf{}.txt".format(self.currentModelID))
            print("Better network found! New network model is: {}".format(self.currentModelID))
            print("Generating a few realistic decks for Network model {}.".format(self.currentModelID))
            self.test_network(None, None, number_of_games = 50)
        else:
            print("Could not find better network!")
        print("{} W / {} T / {} L games against the dummy network as Player 1".format(winsAsP1, tiesAsP1, lossesAsP1))
        print("{} W / {} T / {} L games against the dummy network as Player 2".format(winsAsP2, tiesAsP2, lossesAsP2))
        if not shouldUpdate:
            with open("ttnn/decks/networkdeckchoices{}.txt".format(self.currentModelID), "w") as f:
                f.write(outputString)
            '''with open("cardstats{}.txt".format(self.currentModelID), "w") as f:
                usedIndexes = [False] * len(self.allcards)
                for i in range(len(self.allcards)):
                    bestIndex = -1
                    bestValue = -1
                    for j in range(len(self.allcards)):
                        if (usedIndexes[j]):
                            continue
                        elif self.totalCardsInTests[j] == 0 and bestValue < 0:
                            bestValue = 0
                            bestIndex = j
                        elif self.totalCardsInTests[j] > 0 and bestValue < self.bestCardsInTests[j] / self.totalCardsInTests[j]:
                            bestValue = self.bestCardsInTests[j] / self.totalCardsInTests[j]
                            bestIndex = j

                    usedIndexes[bestIndex] = True
                    f.write("{}:{}% ({}/{})\n".format(self.allcards[bestIndex].name, 100 * bestValue, self.bestCardsInTests[bestIndex], self.totalCardsInTests[bestIndex]))
                    '''
    '''
    Unused method that converts a network output to its card and position and returns it
    '''
    def outputValueToCardAndPos(self, gameboard, outputValue):
        card = None
        if (gameboard.turn % 2 == 0):
            card = gameboard.initialP1Hand[outputValue % 5]
        else:
            card = gameboard.initialP2Hand[outputValue % 5]

        #print (card.name)
        pos = outputValue // 5
        return (card, pos)

    '''
    Internal method of mainmethod (the one that trains the network)
    Returns the hands played if you want to test on the same data as the training data or something (not currently implemented)
    Also writes game logs to file specified by output file
    '''
    def play_games(self, number_of_games = 1000, outputfile = "thousandgames.txt"):
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
                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = j % 2 == 0, hideCards = True)
                #outputFromNN = None

                #print(outputFromNN.shape)
                cardToPlay, posToPlay = (None, 0)
                if j % 2 == 0:
                    if isP1:
                        #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                        cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testnet[0])
                    else:
                        #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                        cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testnet[0])
                else:
                    if isP1:
                        #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                        cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testnet[0])
                    else:
                        #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                        cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testnet[0])
                gameboard.playCard(cardToPlay, posToPlay)
                #print ("Time to complete a turn: {} ms".format(i, int(round(time.time() * 1000)) - startTurnTime))
            gamedata.append((p1, p2, gameboard.score))
            #print ("Time to complete a game: {} ms".format(i, int(round(time.time() * 1000)) - startTime))
            stringedGamesPlayed += gameboard.toFileString(self.allcards, isP1 = isP1) + "\n"
            if i % (number_of_games // 100) == 0:
                print("Played {}% of games".format(i / number_of_games * 100))
        with open(outputfile, "w") as f:
            f.write(stringedGamesPlayed)
        return gamedata

    #loads/reloads the current network from file
    def load(self):
        self.net = [None]
        for i in range(1):
            self.net[0] = tf.keras.models.load_model("ttnn/finalnetwork4tf{}.txt".format(self.currentModelID))
            #self.fivestarnet[0] = tf.keras.models.load_model("ttnn/finalnetwork4tffivestar{}.txt".format(self.currentModelID))
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
        self.p1Hidden[otherInt] = True
        self.p2Hand = [p2[0], p2[1], p2[2], p2[3], p2[4]]
        self.p2Hidden = [False] * 5
        self.p2Hidden[random.randint(0, 4)] = True
        while (self.p2Hidden[otherInt]):
            otherInt = random.randint(0, 4)
        self.p2Hidden[otherInt] = True
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

    #deprecated
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
            secretUnplayedCards = len([value for value in self.initialP2Hand if value in self.p2Hand])
            for i in self.initialP2Hand[0:3]:
                if i.stars == 5:
                    fiveStarWasPlayed = (i in hand)
                    break


        else:
            initialHandLength = len(self.p1Hand)
            secretUnplayedCards = len([value for value in self.initialP1Hand if value in self.p1Hand])
            for i in self.initialP1Hand[0:3]:
                if i.stars == 5:
                    fiveStarWasPlayed = (i in hand)
                    break

        normalUnplayedCards = len(hand)
        totalUnplayedCards = normalUnplayedCards + secretUnplayedCards
        if (contains5Star or fiveStarWasPlayed):
            while(len(hand) < totalUnplayedCards):
                hand.append(Card(["Three Star Wild", 3, 8, 8, 8, 8]))
        else:
            hand.append(Card(["Five Star Wild", 3, 10, 10, 10 ,10]))
            while(len(hand) < totalUnplayedCards):
                hand.append(Card(["Three Star Wild", 3, 8, 8, 8, 8]))
        return hand


    def flipHiddenCards(self, hand, isP1, allcards, normalcards, legendarycards, cardnamedict, fiveStarPrediction = Card(["Five Star Wild", 5, 10, 10, 10, 10])):
        contains5StarHidden = False
        #find out if there is a 5 star in their hand, this is easy to compute (check hand, check board) and always computable so we just cheat to save time
        for i in range(len(hand)):
            if (self.turn % 2 == 0):
                if isP1:
                    if hand[i].stars == 5 and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
                else:
                    if hand[i].stars == 5 and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
            else:
                if isP1:
                    if hand[i].stars == 5 and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
                else:
                    if hand[i].stars == 5 and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
        #make a list of all hidden indexes and replace all hidden cards with our wildcard
        hiddenIndexes = []
        if self.turn % 2 == 0:
            if isP1:
                for i in range(len(hand)):
                    if self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
            else:
                for i in range(len(hand)):
                    if self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
        else:
            if isP1:
                for i in range(len(hand)):
                    if self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
            else:
                for i in range(len(hand)):
                    if self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
        #replace a random hidden card with 5 star wild
        if (contains5StarHidden):
            hand[hiddenIndexes[0]] = fiveStarPrediction
        return hand

    '''
    Yes this is actually necessary. Will thing of a better name / implementation later.
    '''
    def flipHiddenCards2(self, hand, isP1, allcards, normalcards, legendarycards, cardnamedict, fiveStarPrediction = Card(["Five Star Wild", 5, 10, 10, 10, 10])):
        contains5StarHidden = False
        #find out if there is a 5 star in their hand, this is easy to compute (check hand, check board) and always computable so we just cheat to save time
        for i in range(len(hand)):
            if (self.turn % 2 == 0):
                if isP1:
                    if hand[i].stars == 5 and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
                else:
                    if hand[i].stars == 5 and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
            else:
                if isP1:
                    if hand[i].stars == 5 and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
                else:
                    if hand[i].stars == 5 and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        contains5StarHidden = True #whether or not a 5 star is played is simple enough to figure out so well just 'cheat' on our algorithm
                        break
        #make a list of all hidden indexes and replace all hidden cards with our wildcard
        hiddenIndexes = []
        if self.turn % 2 == 0:
            if isP1:
                for i in range(len(hand)):
                    if self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
            else:
                for i in range(len(hand)):
                    if self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
        else:
            if isP1:
                for i in range(len(hand)):
                    if self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
            else:
                for i in range(len(hand)):
                    if self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
                        hiddenIndexes.append(i)
        if (contains5StarHidden):
            hand[hiddenIndexes[random.randint(0, len(hiddenIndexes) - 1)]] = fiveStarPrediction
        return hand

    def getGameboardInputArray(self, allcards, normalcards, legendarycards, cardnamedict, isP1 = False, hideCards = True, fiveStarPrediction = Card(["Five Star Wild", 5, 10, 10, 10, 10])):
        redhand = self.p1Hand.copy() #TODO: dont actually call them this! fix at some point
        bluehand = self.p2Hand.copy()

        if hideCards:
            if (self.turn % 2 == 0):
                if isP1:
                    bluehand = self.flipHiddenCards2(bluehand, isP1, allcards, normalcards, legendarycards, cardnamedict, fiveStarPrediction = fiveStarPrediction)
                else:
                    redhand = self.flipHiddenCards2(redhand, isP1, allcards, normalcards, legendarycards, cardnamedict, fiveStarPrediction = fiveStarPrediction)
            else:
                if isP1:
                    redhand = self.flipHiddenCards2(redhand, isP1, allcards, normalcards, legendarycards, cardnamedict, fiveStarPrediction = fiveStarPrediction)
                else:
                    bluehand = self.flipHiddenCards2(bluehand, isP1, allcards, normalcards, legendarycards, cardnamedict, fiveStarPrediction = fiveStarPrediction)
                #print(len(redhand))


        input = [0] * 145 #input size as described in the network architecture in TripleTriadBot init
        #inputs for gameboardstate
        for k in range(9):
            square = self.gameboard[k]
            if (square is not None):
                arrayIndex = 5 * 10 + 5 * square.pos
                input[arrayIndex + 0] = square.card.left / 10
                input[arrayIndex + 1] = square.card.up / 10
                input[arrayIndex + 2] = square.card.right / 10
                input[arrayIndex + 3] = square.card.down / 10
                input[arrayIndex + 4] = square.owner
        #inputs for known hand cards
        try:
            for k in len(redhand):
                input[5 * k + 0] = redhand[k].left / 10
                input[5 * k + 1] = redhand[k].up / 10
                input[5 * k + 2] = redhand[k].right / 10
                input[5 * k + 3] = redhand[k].down / 10
                input[5 * k + 4] = 0

                input[5 * (k + 19) + 0] = redhand[k].left / 10
                input[5 * (k + 19) + 1] = redhand[k].up / 10
                input[5 * (k + 19) + 2] = redhand[k].right / 10
                input[5 * (k + 19) + 3] = redhand[k].down / 10
                input[5 * (k + 19) + 4] = 0

        except TypeError:
            #nothing needs to be done
            redhand = redhand
        try:
            for k in range(redhand):
                input[5 * (k + 5) + 0] = self.initialP2Hand[k].left / 10
                input[5 * (k + 5) + 1] = self.initialP2Hand[k].up / 10
                input[5 * (k + 5) + 2] = self.initialP2Hand[k].right / 10
                input[5 * (k + 5) + 3] = self.initialP2Hand[k].down / 10
                input[5 * (k + 5) + 4] = 1

                input[5 * (k + 5 + 19) + 0] = self.initialP2Hand[k].left / 10
                input[5 * (k + 5 + 19) + 1] = self.initialP2Hand[k].up / 10
                input[5 * (k + 5 + 19) + 2] = self.initialP2Hand[k].right / 10
                input[5 * (k + 5 + 19) + 3] = self.initialP2Hand[k].down / 10
                input[5 * (k + 5 + 19) + 4] = 1
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

    '''
    Prints a copy of the gameboard in text format
    Adapated from a friend's code for an all open card solver

    Example:

Game Score: -1
    0     1     2
 -------------------
 |  8  |  4  |  3  |
0|8 2 3|8 2 8|9 1 9|
 |  1  |  2  |  8  |
 -------------------
 |  4  |  4  |     |
1|8 2 1|8 2 1|     |
 |  8  |  8  |     |
 -------------------
 |  8  |     |     |
2|4 1 8|     |     |
 |  1  |     |     |
 -------------------

You can see your opponent has: (1) Byakko(u: 1, r:7, d:7, l:7), (2) Hien(u: 10, r:5, d:10, l:2), (3) None, (4) None, (5) None

    '''
    def printgameboard(self):
        sb = "Game Score: {}\n".format(self.score)
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
            redhand = [value for value in self.initialP2Hand if value in self.p2Hand]
            #print(len(self.p2Hand))
        else:
            redhand = [value for value in self.initialP1Hand if value in self.p1Hand]
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

            for i in range(len(self.initialP2Hand)):
                if (self.p2Hidden[i] and self.initialP2Hand[i] in self.p2Hand): cardStrings[self.p2Hand.index(self.initialP2Hand[i])] = "Hidden"
        else:
            for i in range(len(self.initialP1Hand)):
                if (self.p1Hidden[i] and self.initialP1Hand[i] in self.p1Hand): cardStrings[self.p1Hand.index(self.initialP1Hand[i])] = "Hidden"
        #print(secretUnplayedCards)


        print(sb)
        print("You can see your opponent has: (1) {}, (2) {}, (3) {}, (4) {}, (5) {}".format(cardStrings[0], cardStrings[1], cardStrings[2], cardStrings[3], cardStrings[4]))

    def playCard(self, card, pos):
        self.gameboard[pos] = GameSquare(card, self.turn, pos)
        self.attemptFlip(card, pos)
        if (self.turn % 2 == 0):
            #print("You can see your opponent has: (1) {}, (2) {}, (3) {}, (4) {}, (5) {}".format(self.p1Hand[0].name, self.p1Hand[1].name, self.p1Hand[2].name, self.p1Hand[3].name, self.p1Hand[4].name))

            for i in range(len(self.p1Hand)):

                if card == self.p1Hand[i]:
                    self.p1Hand.remove(card)
                    break
        else:
            #print("You can see your opponent has: (1) {}, (2) {}, (3) {}, (4) {}, (5) {}".format(self.p2Hand[0].name, self.p2Hand[1].name, self.p2Hand[2].name, self.p2Hand[3].name, self.p2Hand[4].name))

            for i in range(len(self.p2Hand)):
                if card.name == self.p2Hand[i].name:
                    self.p2Hand.remove(card)
                    break
        self.turn = self.turn + 1
        #print(self.turn)

    '''
    attempt to flip weaker cards adjacent to ours, as according to Triple Triad rules
    '''
    def attemptFlip(self, card, pos):
        #topface
        if (pos > 2):
            if (self.gameboard[pos - 3] is not None):
                if (self.gameboard[pos - 3].card.down < card.up):
                    if self.gameboard[pos - 3].owner != self.turn % 2:
                        self.gameboard[pos - 3].owner = self.turn % 2
                        if (self.turn % 2 == 0):
                            self.score+=1
                        else:
                            self.score-=1
        #bottomface
        if (pos < 6):
            if (self.gameboard[pos + 3] is not None):
                if (self.gameboard[pos + 3].card.up < card.down):
                    if self.gameboard[pos + 3].owner != self.turn % 2:
                        self.gameboard[pos + 3].owner = self.turn % 2
                        if (self.turn % 2 == 0):
                            self.score+=1
                        else:
                            self.score-=1
        #leftface
        if (pos % 3 > 0):
            if (self.gameboard[pos - 1] is not None):
                if (self.gameboard[pos - 1].card.right < card.left):
                    if self.gameboard[pos - 1].owner != self.turn % 2:
                        self.gameboard[pos - 1].owner = self.turn % 2
                        if (self.turn % 2 == 0):
                            self.score+=1
                        else:
                            self.score-=1

        #rightface
        if (pos % 3 < 2):
            if (self.gameboard[pos + 1] is not None):
                if (self.gameboard[pos + 1].card.left < card.right):
                    if self.gameboard[pos + 1].owner != self.turn % 2:
                        self.gameboard[pos + 1].owner = self.turn % 2
                        if (self.turn % 2 == 0):
                            self.score+=1
                        else:
                            self.score-=1

    '''
    Send the gameboard to a text based string for writing to a file

    Example:

    P1 Deck: Papalymo&Yda | Lupin | Coeurlregina | Arenvald | Gosetsu
    P2 Deck: AlexanderPrime | CountEdmontdeFortemps | Alpha | Omega | TozolHuatotl
    Cards played: Gosetsu: 0, 2 | TozolHuatotl: 1, 2 | Arenvald: 2, 2 | CountEdmontdeFortemps: 2, 1 | Coeurlregina: 1, 1 | Omega: 0, 1 | Lupin: 0, 0 | Alpha: 1, 0 | Papalymo&Yda: 2, 0 | P1 Loss
    '''
    def toFileString(self, allCards, isP1 = True):
        fileString = "P1 Deck: "
        for i in range(len(self.initialP1Hand)):
            fileString += self.initialP1Hand[i].name
            if (self.p1Hidden[i]): fileString += "(Hidden)"
            if (i != 4):
                fileString += " | "
        fileString += "\nP2 Deck: "
        for i in range(len(self.initialP1Hand)):
            fileString += self.initialP2Hand[i].name
            if (self.p2Hidden[i]): fileString += "(Hidden)"
            if (i != 4):
                fileString += " | "
        fileString += "\nCards played: "
        for i in range(9):
            for j in range(9):
                if (self.gameboard[j] is not None):
                    if (i == self.gameboard[j].turnPlayed):
                        fileString += self.gameboard[j].card.name + ": " + str(self.gameboard[j].pos % 3) + ", " + str(self.gameboard[j].pos // 3)
                        fileString += " | "
        if isP1:
            if (self.score > 0):
                fileString += "P1 Win"
            elif (self.score == 0):
                fileString += "Tie"
            else:
                fileString += "P1 Loss"
        else:
            if (self.score < 0):
                fileString += "P2 Win"
            elif (self.score == 0):
                fileString += "Tie"
            else:
                fileString += "P2 Loss"
        return fileString
class GameSquare(object):
    def __init__(self, card, turn, pos):
        self.card = card
        self.turnPlayed = turn
        self.pos = pos
        self.owner = turn % 2

    def clone(self):
        return GameSquare(self.card, self.turnPlayed, self.pos)

class TreeNode(object):
    """docstring for TreeNode."""

    def __init__(self, data, history, isP1, turn = 0):
        super(TreeNode, self).__init__()
        self.choice = history[0] #between 0 and 44
        self.turn = turn
        self.wins = 0
        self.isP1 = isP1
        if isP1:
            if self.turn % 2 == 0:
                if data[0] > 0: self.wins = 1
            else:
                if data[2] > 0: self.wins = 1
        else:
            if self.turn % 2 == 0:
                if data[2] > 0: self.wins = 1
            else:
                if data[0] > 0: self.wins = 1
        self.total = 1
        if len(history) == 1:
            self.children = []
        else:
            self.children = [TreeNode(data, history[1:], isP1, turn + 1)]

    def getChild(self, choice):
        for i in range(len(self.children)):
            if self.children[i].choice == choice:
                return self.children[i]
        return None
    def addChild(self, data, history):
        #check if this child exists
        if self.isP1:
            if self.turn % 2 == 0:
                if data[0] > 0: self.wins = 1
            else:
                if data[2] > 0: self.wins = 1
        else:
            if self.turn % 2 == 0:
                if data[2] > 0: self.wins = 1
            else:
                if data[0] > 0: self.wins = 1
        self.total += 1
        for i in range(len(self.children)):
            if self.children[i].choice == history[0]:

                self.children[i].addChild(data, history[1:])
                return
        self.children.append(TreeNode(data, history, self.isP1, self.turn + 1))

    def doesChildExist(self, history):
        if (len(self.children) == 0): return True
        if len(history) == 0: return True
        for i in range(len(self.children)):
            if self.children[i].choice == history[0]:
                return self.children[i].doesChildExist(history[1:])
        return False

    def isDead(self):
        if (len(self.children) == 0): return True
        totalDead = 0
        for i in range(len(self.children)):
            #print (self.children[i])
            if self.children[i].isDead(): totalDead +=1
        return totalDead >= ((9 - self.turn) + 1) // 2 * (9 - self.turn)
