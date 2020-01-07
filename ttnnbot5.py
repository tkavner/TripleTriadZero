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

import ttnnbot4

normal, legendary, allcards, cardnamedict = ttnnbot4.make_card_data()
with open("outputdecks.txt") as f:
    carddatarawunsplit = f.read()
carddataraw = carddatarawunsplit.split("\n")


def get_bot_deck(deckID):
    if (deckID > len(carddataraw)): deckID = deckID % len(carddataraw)
    line = carddataraw[deckID]
    cardNames = line[line.index(":") + 1:].replace(" ", "").replace(":", "").split("|")
    return [allcards[cardnamedict[cardNames[0]]], allcards[cardnamedict[cardNames[1]]], allcards[cardnamedict[cardNames[2]]], allcards[cardnamedict[cardNames[3]]], allcards[cardnamedict[cardNames[4]]]]

                    # 0      1      2      3      4      5      6      7      8
turnsToDeadArray = [9 * 5, 8 * 5, 7 * 4, 6 * 4, 5 * 3, 4 * 3, 3 * 2, 2 * 2, 1 * 1]

def turnsToDead(turn):
    return turnsToDeadArray[turn]

def sort_tuple(val):
    return val[0]


class TripleTriadBot5(object):

    def __init__(self, currentModelID = 0, cardfile = 'cardslimited.txt'):

        self.currentModelID = currentModelID
        self.normal, self.legendary, self.allcards, self.cardnamedict = ttnnbot4.make_card_data(cardfile = cardfile)
        self.playerrecord = [0,0,0]

        self.choiceMaker = ChoiceMaker(currentModelID, cardfile)
        self.cardSelector = CardSelector(currentModelID, cardfile)
        self.cardPredictor = CardPredictor(currentModelID, cardfile)

        self.testChoiceMaker = ChoiceMaker(currentModelID, cardfile)
        self.testCardSelector = CardSelector(currentModelID, cardfile)
        self.testCardPredictor = CardPredictor(currentModelID, cardfile)

        if (currentModelID == -1):
            self.currentModelID = 0
            tf.keras.models.save_model(self.choiceMaker.model, "ttnn5/finalnetwork5m{}.txt".format(self.currentModelID))
            tf.keras.models.save_model(self.cardSelector.model, "ttnn5/finalnetwork5s{}.txt".format(self.currentModelID))
            tf.keras.models.save_model(self.cardPredictor.model, "ttnn5/finalnetwork5p{}.txt".format(self.currentModelID))
        self.load()

        random.seed(datetime.datetime.now()) #'quality' method of seeding random hand generation
        self.bestCardsInTests = [0] * len(self.allcards)
        self.totalCardsInTests = [0] * len(self.allcards)

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

    '''
    Call this to train the network
    Optional parameters are depretacted
    '''
    def train_selector(self, iterations = 200, numgames=100):
        for i in range(1000):
            nplist = self.play_selector_games("ttnn5/games/thousandgames{}.txt".format(i), number_of_games = numgames, train_selector = train_selector)
            print("Processing Selector")
            datasetIn, datasetOut = nplist
            self.cardSelector.model.fit(datasetIn, datasetOut, epochs=iterations)
            tf.keras.models.save_model(self.cardSelector.model, "ttnn5/finalnetwork5s{}.txt".format(self.currentModelID + 1))
            print("Saved selector network to file!")


    def load(self):
        del self.choiceMaker.model
        del self.cardSelector.model
        del self.cardPredictor.model
        self.choiceMaker.model = tf.keras.models.load_model("ttnn5/finalnetwork5m{}.txt".format(self.currentModelID))
        self.cardSelector.model = tf.keras.models.load_model("ttnn5/finalnetwork5s{}.txt".format(self.currentModelID))
        self.cardPredictor.model = tf.keras.models.load_model("ttnn5/finalnetwork5p{}.txt".format(self.currentModelID))

    def MCTS2Top(self, gameboard, canSeeWholeHand, networkToPredictWith, timeToEstimate = 333, training = False, initialNodes = 6, testPredictor = None):
        '''
        timeToEstimateEntireTurn = turnsToDead(gameboard.turn) * 7 #we can get about one turn per 7.35 milliseconds. so this is a very rough estimation
        if timeToEstimate > timeToEstimateEntireTurn / 2: #we want to set a maximum if there's say... only 24 ways to play the game from this turn possible
            timeToEstimate = timeToEstimateEntireTurn / 2
        if timeToEstimate < 100:
            timeToEstimate = 100
        '''
        startTime = int(round(time.time() * 1000))
        #we (the predictor) are p1 if we can see our entire hand on an even turn or if we cannot see our entire hand on a odd (p2) turn
        isP1 = (canSeeWholeHand and gameboard.turn % 2 == 0) or ((not canSeeWholeHand) and gameboard.turn % 2 == 1)

        hiddenCardPredictions = None

        if testPredictor is not None:
            hiddenCardPredictions = testPredictor.predictHiddenCards(gameboard, self.allcards, self.normal, self.legendary, self.cardnamedict, isP1)
        else:
            hiddenCardPredictions = self.cardPredictor.predictHiddenCards(gameboard, self.allcards, self.normal, self.legendary, self.cardnamedict, isP1)
        inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, hiddenCardPredictions = hiddenCardPredictions)
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
                    redhand = gameboard.flipHiddenCards(redhand, False, self.allcards, self.normal, self.legendary, self.cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
                    card = redhand[validIndexes[i] % 5]
                else:
                    redhand = [value for value in gameboard.initialP2Hand]
                    #if we cannot see the whole hand and its turn 1, 3, 5, or 7, then we are p1
                    redhand = gameboard.flipHiddenCards(redhand, True, self.allcards, self.normal, self.legendary, self.cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
                    card = redhand[validIndexes[i] % 5]

            gbCopy.playCard(card, pos)
            gbCopies.append(gbCopy)

            if (not canSeeWholeHand):
                #its important that the hidden card actually gets played from the original hand, else it'll be available next time! We can check for this by checking if a card was removed from the p1/p2Hand by comparing to redhand's length
                #we do need to note that the turn incremented when the card was played however.
                if gbCopy.turn % 2 == 1:
                    if len(redhand) == len(gbCopy.p1Hand):
                        gbCopy.p1Hand.remove(gbCopy.initialP1Hand[validIndexes[i] % 5])
                else:
                    if len(redhand) == len(gbCopy.p2Hand):
                        gbCopy.p2Hand.remove(gbCopy.initialP2Hand[validIndexes[i] % 5])



        for i in range(len(validIndexes)):
            data, card, pos, history = self.MCTS2(gbCopies[i], 6, canSeeWholeHand, networkToPredictWith, training = training, hiddenCardPredictions = hiddenCardPredictions, desiredValidIndexes = 1, tree = None)
            history = [validIndexes[i]] + history
            if (len(history) != 0):
                topNodes.append(TreeNode(data, history, isP1, gbCopies[i].turn))
            else:
                for j in range(9):
                    if gbCopies[i].gameboard[j].turnPlayed == 8:
                        return (gbCopies[i].gameboard[j].card, gbCopies[i].gameboard[j].pos)
                print("I'm still here and I'm still alive.")

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
                nodeValue = topNodes[i].wins / topNodes[i].total + (math.log(totalNodes) / topNodes[i].total * 1.5) ** 0.5
                if not topNodes[i].isDead():
                    if nodeValue > bestNodeValue:

                        bestNode = i
                        bestNodeValue = nodeValue
                else:
                    return self.outputValueToCardAndPos(gameboard, topNodes[i].choice) #if we have a node with no more choices, we choose that one, it means the MCTS has had too much time to calculate. The fact it got through the whole tree indicates great confidence in the node so we choose that one.
            if (bestNode < 0): #this means we've exhausted our search, we're trying to avoid this but its gunna happen
                searchExhausted = True #if this happens we need a new metric to decide the best node
                break
            #we use "not canSeeWholeHand" because we're talking about the turn after this one
            data, card, pos, history = self.MCTS2(gbCopies[bestNode], 6, not canSeeWholeHand, networkToPredictWith, training = training, hiddenCardPredictions = hiddenCardPredictions, desiredValidIndexes = 1, tree = topNodes[bestNode])
            topNodes[bestNode].addChild(data, history)
        bestNode = -100000
        bestNodeValue = -100000
        nodeTotal = 0
        for i in range(len(topNodes)):
            nodeValue = topNodes[i].total
            if searchExhausted:
                nodeValue = topNodes[i].wins / topNodes[i].total
            if nodeValue > bestNodeValue:
                bestNode = i
                bestNodeValue = nodeValue
            elif nodeValue == bestNodeValue:
                if topNodes[i].wins / topNodes[i].total > topNodes[bestNode].wins / topNodes[bestNode].total:
                    bestNode = i
                    bestNodeValue = nodeValue
            nodeTotal += topNodes[i].total
        return self.outputValueToCardAndPos(gameboard, topNodes[bestNode].choice)

    def MCTS2(self, gameboard, turnsToCheck, canSeeWholeHand, networkToPredictWith, training = False, shouldBeRandom = False, topLayer = True, layerOutput = None, hiddenCardPredictions = None, desiredValidIndexes = 1, playToTie = False, tree = None, testPredictor = None):
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

            inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, hiddenCardPredictions = hiddenCardPredictions)
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
        if len(validIndexes) == 0:
            gameboard.printgameboard()
            print(b)
            print(indexesInOrder)
            for i in range(45):
                card = None

                if (gameboard.turn % 2 == 0):
                    card = gameboard.initialP1Hand[i % 5]
                else:
                    card = gameboard.initialP2Hand[i % 5]


                pos = i // 5

                print(gameboard.isValidMove(card, pos))
            print("Children of tree: {}".format(len(tree.children)))

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
                    redhand = gameboard.flipHiddenCards(redhand, False, self.allcards, self.normal, self.legendary, self.cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
                    card = redhand[validIndexes[i] % 5]


                else:
                    redhand = [value for value in gameboard.initialP2Hand]
                    #conversely if we cannot see the whole hand and its an odd turn, we (the predictor) must be p1
                    redhand = gameboard.flipHiddenCards(redhand, True, self.allcards, self.normal, self.legendary, self.cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
                    card = redhand[validIndexes[i] % 5]
                    #its important that the hidden card actually gets played from the original hand, else it'll be available next time!

            gbCopy.playCard(card, pos)
            if (not canSeeWholeHand):
                #its important that the hidden card actually gets played from the original hand, else it'll be available next time! We can check for this by checking if the redhand is the same length as the p1/p2 hand
                #we do need to note that the turn incremented when the card was played however.
                if gbCopy.turn % 2 == 1:
                    if len(redhand) == len(gbCopy.p1Hand):
                        gbCopy.p1Hand.remove(gbCopy.initialP1Hand[validIndexes[i] % 5])
                else:
                    if len(redhand) == len(gbCopy.p2Hand):
                        gbCopy.p2Hand.remove(gbCopy.initialP2Hand[validIndexes[i] % 5])


            gbCopies.append(gbCopy)

        history = [validIndexes[0]]
        if turnsToCheck > 1:
            #list of all inputs for the neural network
            inputsForNN = []
            #we now create the new inputs for the next depth of the MCTS we do it this way because it saves computation time in Tensorflow

            for i in range(len(validIndexes)):
                #inputsFor5SNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1)
                inputsForNN += gbCopies[i].getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = isP1, hiddenCardPredictions = hiddenCardPredictions)

            #feed data to our network, we reshape to make sure its the right format for both the input and the output
            outputsFromNN = np.reshape(networkToPredictWith.predict(np.reshape(np.array(inputsForNN, order='F', ndmin = 4), (-1, 5, 29, 1))), (len(validIndexes), 45))

            #now we ship off the values to next layer of the MCTS
            for i in range(len(validIndexes)):
                hiddenCardPred = hiddenCardPredictions
                if tree is not None:

                    mctsdata, card, pos, childHistory = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, layerOutput = outputsFromNN[i], hiddenCardPredictions = hiddenCardPred, desiredValidIndexes = desiredValidIndexes, tree = tree.getChild(validIndexes[i]))
                    history += childHistory
                    scores[i] = mctsdata
                else:
                    mctsdata, card, pos, childHistory = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, layerOutput = outputsFromNN[i], hiddenCardPredictions = hiddenCardPred, desiredValidIndexes = desiredValidIndexes, tree = None)
                    history += childHistory
                    scores[i] = mctsdata


        else:
            #then next turn will not be using its output data from the neural network, so we simply have it skip calculating the final turn
            for i in range(len(validIndexes)):
                if tree is not None:
                    mctsdata, card, pos, childHistory  = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, hiddenCardPredictions = hiddenCardPredictions, tree = tree.getChild(validIndexes[i]))
                    history += childHistory
                    scores[i] = mctsdata
                else:
                    mctsdata, card, pos, childHistory  = self.MCTS2(gbCopies[i], turnsToCheck - 1, not canSeeWholeHand, networkToPredictWith, training = training, shouldBeRandom = shouldBeRandom, topLayer = False, hiddenCardPredictions = hiddenCardPredictions, tree = None)
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

    def outputValueToCardAndPos(self, gameboard, outputValue):
        card = None
        if (gameboard.turn % 2 == 0):
            card = gameboard.initialP1Hand[outputValue % 5]
        else:
            card = gameboard.initialP2Hand[outputValue % 5]

        #print (card.name)
        pos = outputValue // 5
        return (card, pos)


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

                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, hideCards = True)
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
                outputForNN = [0] * 8
                hiddenCards = 0
                inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict)
                if (j % 2 == 0):
                    card = None

                    for i in range(len(gameboard.initialP1Hand)):
                        if (gameboard.initialP1Hand[i] in gameboard.p1Hand):
                            card = gameboard.initialP1Hand[i]
                            if (gameboard.p1Hidden[i]):
                                cardSum = (card.left + card.up + card.right + card.down)
                                outputForNN[4 * hiddenCards + 0] = card.left / cardSum
                                outputForNN[4 * hiddenCards + 1] = card.up / cardSum
                                outputForNN[4 * hiddenCards + 2] = card.right / cardSum
                                outputForNN[4 * hiddenCards + 3] = card.down / cardSum
                                hiddenCards += 1
                        if (hiddenCards > 1):
                            break
                    if outputForNN[0] == 0:
                        continue

                else:
                    card = None
                    for i in range(len(gameboard.initialP2Hand)):
                        if (gameboard.initialP2Hand[i] in gameboard.p2Hand):
                            card = gameboard.initialP2Hand[i]
                            if (gameboard.p2Hidden[i]):
                                cardSum = (card.left + card.up + card.right + card.down)
                                outputForNN[4 * hiddenCards + 0] = card.left / cardSum
                                outputForNN[4 * hiddenCards + 1] = card.up / cardSum
                                outputForNN[4 * hiddenCards + 2] = card.right / cardSum
                                outputForNN[4 * hiddenCards + 3] = card.down / cardSum
                                hiddenCards += 1
                        if (hiddenCards > 1):
                            break

                    if outputForNN[0] == 0:
                        continue


                gameboard.playCard(self.allcards[self.cardnamedict[cardName]], pos)

                totalValueOfOutput = sum(outputForNN)

                for i in range(len(outputForNN)):
                    outputForNN[i] /= totalValueOfOutput

                if (nplistIn[0] == None):
                    nplistIn[0] = inputForNN
                    nplistOut[0] = outputForNN
                else:
                    nplistIn[0] += inputForNN
                    nplistOut[0] += outputForNN
        for j in range(1):
            npinput = np.reshape(np.array(nplistIn[j], order='F', ndmin = 4), [(len(nplistIn[j]) // 145), 5, 29, 1])
            npoutput = np.reshape(np.array(nplistOut[j], order='F', ndmin = 4), [(len(nplistOut[j]) // 8), 1, 8])
            nplist[0] = (npinput, npoutput)
        print(npinput.shape)
        return nplist

    def play_selector_games(self, outputfile, number_of_games = 100, train_selector = True):
        finalFile = ""
        inputsList = [None] * (number_of_games)
        outputsList = [None] * (number_of_games)

        totalCorrect = 0

        originalHand = self.generate_random_hand()
        replacedHand = originalHand.copy()

        for i in range(number_of_games):


            points = 0 #positive indicates a win record for replaced hand, negative for the original hand
            replacingCard = None
            originalCard = None
            cardToReplace = 0

            if (train_selector):
                cardToReplace = random.randint(0, 4)
                originalCard = replacedHand[cardToReplace]
                replacingCard = self.cardSelector.get_better_card_for_hand(replacedHand, cardToReplace, self.allcards, training = train_selector)
                replacedHand.remove(replacedHand[cardToReplace])
                replacedHand.append(replacingCard)

            else:
                if random.randint(0, 100) > 20:
                    replacedHand = self.generate_random_hand()
                    originalHand = self.generate_random_hand()
                    cardToReplace = random.randint(0, 4)
                    originalCard = replacedHand[cardToReplace]
                    replacingCard = self.cardSelector.get_better_card_for_hand(replacedHand, cardToReplace, self.allcards, training = False)
                    replacedHand.remove(replacedHand[cardToReplace])
                    replacedHand.append(replacingCard)

                    cardToReplace = random.randint(0, 4)
                    originalCard = originalHand[cardToReplace]
                    replacingCard = self.testCardSelector.get_better_card_for_hand(originalHand, cardToReplace, self.allcards, training = False)
                    originalHand.remove(originalHand[cardToReplace])
                    originalHand.append(replacingCard)

                    replacingCard = replacedHand[0]
                else:
                    replacedHand = self.generate_random_hand()
                    originalHand = self.generate_random_hand()
                    replacingCard = replacedHand[0]
                    originalCard = originalHand[0]


            random.shuffle(replacedHand)
            random.shuffle(originalHand)

            print("{} {} {} {} {}".format(replacedHand[0].name, replacedHand[1].name, replacedHand[2].name, replacedHand[3].name, replacedHand[4].name))
            print("{} {} {} {} {}".format(originalHand[0].name, originalHand[1].name, originalHand[2].name, originalHand[3].name, originalHand[4].name))

            numGamesPlayed = 0

            while points == 0 or (numGamesPlayed < 2 and train_selector):

                numGamesPlayed+=1

                opponentHand = self.generate_random_hand()

                #combat opponentHand and replacedHand
                isP1 = random.randint(0,1)==0
                p1 = None
                p2 = None
                if isP1:
                    p1 = replacedHand
                    p2 = opponentHand
                else:
                    p2 = replacedHand
                    p1 = opponentHand
                gameboard = Gameboard(p1.copy(), p2.copy())
                for j in range(9):
                    cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.choiceMaker.model)
                    gameboard.playCard(cardToPlay, posToPlay)

                if isP1:
                    if gameboard.score > 0:
                        points += 1
                    elif gameboard.score < 0:
                        points += -1
                else:
                    if gameboard.score < 0:
                        points += 1
                    elif gameboard.score > 0:
                        points += -1

                #combat opponentHand and originalHand
                if isP1:
                    p1 = opponentHand
                    p2 = originalHand
                else:
                    p2 = opponentHand
                    p1 = originalHand
                gameboard = Gameboard(p1.copy(), p2.copy())
                for j in range(9):
                    cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.choiceMaker.model)
                    gameboard.playCard(cardToPlay, posToPlay)

                if isP1:
                    if gameboard.score > 0:
                        points += -1
                    elif gameboard.score < 0:
                        points += 1
                else:
                    if gameboard.score < 0:
                        points += -1
                    elif gameboard.score > 0:
                        points += 1

                finalFile += gameboard.toFileString(self.allcards, isP1 = isP1) + "\n"

                random.shuffle(replacedHand)
                random.shuffle(originalHand)

            correctSelection = None
            if points < 0:
                correctSelection = originalCard
                originalHand = self.generate_random_hand() #reset the decks if we can't guess correctly TODO: why?
                replacedHand = originalHand.copy()
            else:
                totalCorrect += 1
                correctSelection = replacingCard
                originalHand = replacedHand.copy()

            if not train_selector:
                if random.randint(0, 100) > 20:
                    replacedHand = self.generate_random_hand()
                    originalHand = self.generate_random_hand()
                    cardToReplace = random.randint(0, 4)
                    originalCard = replacedHand[cardToReplace]
                    replacingCard = self.cardSelector.get_better_card_for_hand(replacedHand, cardToReplace, self.allcards, training = False)
                    replacedHand.remove(replacedHand[cardToReplace])
                    replacedHand.append(replacingCard)

                    cardToReplace = random.randint(0, 4)
                    originalCard = originalHand[cardToReplace]
                    replacingCard = self.testCardSelector.get_better_card_for_hand(originalHand, cardToReplace, self.allcards, training = False)
                    originalHand.remove(originalHand[cardToReplace])
                    originalHand.append(replacingCard)

                    replacingCard = replacedHand[0]
                else:
                    replacedHand = self.generate_random_hand()
                    originalHand = self.generate_random_hand()
                    replacingCard = replacedHand[0]
                    originalCard = originalHand[0]

            correctCardNormalizer = correctSelection.left +  correctSelection.up +  correctSelection.right +  correctSelection.down
            correctOutput = [0] * 4
            correctOutput[0] = correctSelection.left / correctCardNormalizer
            correctOutput[1] = correctSelection.up / correctCardNormalizer
            correctOutput[2] = correctSelection.right / correctCardNormalizer
            correctOutput[3] = correctSelection.down / correctCardNormalizer

            outputsList[i] = correctOutput
            #todo: figure out why card copies are getting put in the decks. also change the judgment back to a competition.
            input = [0] * 16
            increment = 0
            for k in range(len(originalHand)):
                if increment != cardToReplace:
                    input[4 * increment + 0] = originalHand[k].left / 10
                    input[4 * increment + 1] = originalHand[k].up / 10
                    input[4 * increment + 2] = originalHand[k].right / 10
                    input[4 * increment + 3] = originalHand[k].down / 10
                    increment += 1


            inputsList[i] = input




            print("Played {}% of games".format((i + 1) / number_of_games * 100))
            print("Successfully chose {} better hands in {} hands.".format(totalCorrect, number_of_games))

        npinput = np.reshape(np.array(inputsList, order='F', ndmin = 4), [number_of_games, 4, 4, 1])
        npoutput = np.reshape(np.array(outputsList, order='F', ndmin = 4), [number_of_games, 1, 4])

        print("Successfully chose {} better hands in {} hands.".format(totalCorrect, number_of_games))

        with open(outputfile, "w") as f:
            f.write(finalFile)
        return (npinput, npoutput)


    '''
    Plays a game with identical hands, and one card removed from the observed player, writes games to a file, returns gamedata in the form of the hands, score, the replaced card index, and which player had the replaced card, as well as the stringed file data.
    Meant for use for training the CardSelector network
    '''
    def play_selector_game(self, originalHand, train_selector = False):
        stringedGamesPlayed = ""
        gamedata = []
        replacedHand = None
        if (train_selector):
            cardToReplace = random.randint(0, 4)
            replacingCard = self.cardSelector.get_better_card_for_hand(originalHand, cardToReplace, self.allcards)
            replacedHand = originalHand.copy()
            replacedHand.remove(originalHand[cardToReplace])
            replacedHand.append(replacingCard)
        else:
            replacedHand = self.generate_random_hand()
        p1 = originalHand
        p2 = originalHand

        totalWins = 0
        totalTies = 0
        totalLosses = 0
        i = 0
        while i < 1:
            i+=1
            if i > 10:
                originalHand = self.generate_random_hand()
                cardToReplace = random.randint(0, 4)
                replacingCard = self.cardSelector.get_better_card_for_hand(originalHand, cardToReplace, self.allcards)
                replacedHand = originalHand.copy()
                replacedHand.remove(originalHand[cardToReplace])
                replacedHand.append(replacingCard)
                i = 0
            startTime = int(round(time.time() * 1000))
            isP1 = random.randint(0,1)==0 #determines which player has the replaced card (called the observed hand)


            p1 = originalHand #reset the hands before replacing cards
            p2 = originalHand

            if (isP1): #replace one card in the hand we are observing
                p1 = replacedHand
            else:
                p2 = replacedHand

            gameboard = Gameboard(p1.copy(), p2.copy())
            for j in range(9):
                #startTurnTime = int(round(time.time() * 1000))
                cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testChoiceMaker.model)
                gameboard.playCard(cardToPlay, posToPlay)
                #print ("Time to complete a turn: {} ms".format(i, int(round(time.time() * 1000)) - startTurnTime))
            gamedata.append((p1.copy(), p2.copy(), gameboard.score))
            if isP1:
                if gameboard.score > 0:
                    totalWins += 1
                elif gameboard.score < 0:
                    totalLosses += 1
            else:
                if gameboard.score < 0:
                    totalWins += 1
                elif gameboard.score > 0:
                    totalLosses += 1
            if gameboard.score == 0:
                totalTies = 1
            cardToReplace = random.randint(0, 4)
            replacingCard = self.cardSelector.get_better_card_for_hand(originalHand, cardToReplace, self.allcards)
            replacedHand = originalHand.copy()
            replacedHand.remove(originalHand[cardToReplace])
            replacedHand.append(replacingCard)
            #print ("Time to complete a game: {} ms".format(i, int(round(time.time() * 1000)) - startTime))
            stringedGamesPlayed += gameboard.toFileString(self.allcards, isP1 = isP1) + "\n"

        with open("ttnn5/games/thousandgames{}.txt", "w") as f:
            f.write(stringedGamesPlayed)

        return (gamedata, stringedGamesPlayed, totalWins, totalTies, totalLosses, originalHand, originalHand[cardToReplace], replacingCard, cardToReplace)

    def trainPredictor(self, iterations = 5, gameNumber = 0):
        nplist = self.formatGameFiveStar("ttnn5/games/thousandgames{}.txt".format(gameNumber))
        datasetIn, datasetOut = nplist[0]
        self.cardPredictor.model.fit(datasetIn, datasetOut, epochs=iterations)

        tf.keras.models.save_model(self.cardPredictor.model, "ttnn5/finalnetwork5p{}.txt".format(self.currentModelID + 1))

    def main_method(self, number_of_games = 300, iterations = 100, train_selector = True):
        for i in range(1000):
            self.load()


            handdata = self.play_selector_games(number_of_games = number_of_games, outputfile = "ttnn5/games/thousandgames{}.txt".format(i), train_selector = False)
            nplist = self.formatGame("ttnn5/games/thousandgames{}.txt".format(i))

            for j in range(1):
                print("Processing network {}".format(j))
                datasetIn, datasetOut = nplist[0]
                self.choiceMaker.model.fit(datasetIn, datasetOut, epochs=iterations)
                self.trainPredictor(iterations = iterations, gameNumber = i)
                tf.keras.models.save_model(self.choiceMaker.model, "ttnn5/finalnetwork5m{}.txt".format(self.currentModelID + 1))
                print("Saved Choice Maker Network to file!")
            handdata = self.play_selector_games(number_of_games = number_of_games, outputfile = "ttnn5/games/thousandgames{}.txt".format(i), train_selector = True)
            print("Processing Selector")
            datasetIn, datasetOut = handdata
            self.cardSelector.model.fit(datasetIn, datasetOut, epochs=iterations)
            tf.keras.models.save_model(self.cardSelector.model, "ttnn5/finalnetwork5s{}.txt".format(self.currentModelID + 1))
            print("Saved selector network to file!")
            self.test_network(handdata, number_of_games = number_of_games // 2, shouldUpdate = True, train_selector = train_selector)

    def test_network(self, handdata, directory = "nn1/", number_of_games = 1000, shouldUpdate = True, train_selector = False):
        stringedGamesPlayed = ""

        winsAsP1 = 0
        winsAsP2 = 0
        tiesAsP1 = 0
        tiesAsP2 = 0
        lossesAsP1 = 0
        lossesAsP2 = 0
        outputString = ""
        numdeckstoTest = 1
        for i in range(number_of_games):
            if not shouldUpdate: print("Starting deck {} processing...".format(i))
            isP1 = random.randint(0,1)==0
            originalHand = self.generate_random_hand()
            replacedHand = self.generate_random_hand()
            for i in range(1):
                cardToReplace = random.randint(0, 4)
                replacingCard = self.cardSelector.get_better_card_for_hand(replacedHand, cardToReplace, self.allcards, training = False)
                replacedHand.remove(replacedHand[cardToReplace])
                replacedHand.append(replacingCard)

                cardToReplace = random.randint(0, 4)
                originalCard = originalHand[cardToReplace]
                replacingCard = self.testCardSelector.get_better_card_for_hand(originalHand, cardToReplace, self.allcards, training = False)
                originalHand.remove(originalHand[cardToReplace])
                originalHand.append(replacingCard)

            if isP1:
                p1 = replacedHand.copy()
                p2 = originalHand.copy()
            else:
                p1 = originalHand.copy()
                p2 = replacedHand.copy()

            wins = 0
            for i1 in range(numdeckstoTest):
                random.shuffle(p1)
                random.shuffle(p2)
                gameboard = Gameboard(p1.copy(), p2.copy())
                for j in range(9):
                    cardToPlay, posToPlay = (None, 0)

                    if j % 2 == 0:
                        if isP1:
                            #outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.choiceMaker.model)
                        else:
                            #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testChoiceMaker.model, testPredictor = self.testCardPredictor)

                    else:
                        if isP1:
                            #outputFromNN = self.testnet[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.testChoiceMaker.model, testPredictor = self.testCardPredictor)

                        else:
                            #outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                            cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.choiceMaker.model)

                    #print(outputFromNN.shape)
                    gameboard.playCard(cardToPlay, posToPlay)


                if isP1:


                    if gameboard.score > 0:
                        wins += 1
                        winsAsP1 +=1

                    elif gameboard.score == 0:
                        tiesAsP1 +=1
                    else:
                        lossesAsP1 += 1
                else:
                    if (gameboard.score < 0):
                        wins += 1
                        winsAsP2 +=1
                    elif gameboard.score == 0:
                        tiesAsP2 +=1
                    else:
                        lossesAsP2 += 1
                if shouldUpdate:
                    print(gameboard.toFileString(self.allcards, isP1 = isP1))



        if (winsAsP2 + winsAsP1) / (winsAsP2 + winsAsP1 + lossesAsP1 + lossesAsP2) > 0.535 and shouldUpdate:
            self.currentModelID += 1
            del self.testChoiceMaker.model
            del self.testCardSelector.model
            del self.testCardPredictor.model
            del self.testChoiceMaker
            del self.testCardSelector
            del self.testCardPredictor
            self.testChoiceMaker = ChoiceMaker(self.currentModelID)
            self.testCardSelector = CardSelector(self.currentModelID)
            self.testCardPredictor = CardPredictor(self.currentModelID)
            print("Better network found! New network model is: {}".format(self.currentModelID))
        else:
            print("Could not find better network!")
        print("{} W / {} T / {} L games against the dummy network as Player 1".format(winsAsP1, tiesAsP1, lossesAsP1))
        print("{} W / {} T / {} L games against the dummy network as Player 2".format(winsAsP2, tiesAsP2, lossesAsP2))

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
        #p1 = get_bot_deck(self.currentModelID)
        #p2 = get_bot_deck(self.currentModelID)
        random.shuffle(p1)
        random.shuffle(p2)


        for i in range(1):
            cardToReplace = random.randint(0, 4)
            if youAreP1:
                replacingCard = self.cardSelector.get_better_card_for_hand(p2, cardToReplace, self.allcards)
                p2.remove(p2[cardToReplace])
                p2.append(replacingCard)
            else:
                replacingCard = self.cardSelector.get_better_card_for_hand(p1, cardToReplace, self.allcards)
                p1.remove(p1[cardToReplace])
                p1.append(replacingCard)


        gameboard = Gameboard(p1, p2)

        for j in range(9):
            #inputForNN = gameboard.getGameboardInputArray(self.allcards, self.normal, self.legendary, self.cardnamedict, isP1 = not youAreP1)
            #outputFromNN = None
            originalHand = self.generate_random_hand()


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
                    cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.choiceMaker.model)
                    #print("Odds of bot victory: {}%".format(int(100 * mctsResults[0] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    #print("Odds of bot tie: {}%".format(int(100 * mctsResults[1] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    #print("Odds of bot loss: {}%".format(int(100 * mctsResults[2] / (mctsResults[0] + mctsResults[1] + mctsResults[2]))))
                    print(cardToPlay.name)
                    gameboard.playCard(cardToPlay, posToPlay)

            else:
                if youAreP1:
                    print("Thinking...")
                    #outputFromNN = self.net[0].predict(np.reshape(np.array(inputForNN, order='F', ndmin = 4), (-1, 5, 29, 1)))
                    cardToPlay, posToPlay = self.MCTS2Top(gameboard.clone(), True, self.choiceMaker.model)
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

'''
    A network that when given an input for 4 cards in a hand + the current card being used will attempt to find a better card to be used by the hands
    This is achieved by feeding the a Tensorflow model inputs in the order of the cards in the hand, with the card that is be reselected replaced with zeros and added to a special sixth slot

    In practice if we have a hand with cards c1... c5, and we want to find a better card to match c3, we would feed the inputs as follows:

    c1, c2, zero, c4, c5, c3

    To match a better than c4 we would do:

    c1 c2 c3 zero c5 c4

    We train the network to be able to find better cards by having it give back the ratios of stats (normalized stats) that it believes a winning card would have,
    and matching it to the card from our list that most properly represents those values

    We then check for the closest card that the hand can hold that has these stat ratios. If it believes no better card can be found, it should return the original card's ratios.

    We use 24 inputs in this fashion instead of 16 inputs (of just the four cards to be paired with) because of how we are testing and training the network.
    Testing is done by using a pair of networks, one CardPredictor and one ChoiceMaker, which need to have learned enough to play the game at a basic level or higher.
    A sufficiently large odd number (11 as of the writing of this) of games is played with the original hand versus the new hand. This reduces the burden of randomization on being player 1 versus player 2.
    The hand winning the majority of the matches is determined to be the correct data choice, and this data is used to update the selector's model.

    Because of the nature of using the CardPredictor and ChoiceMaker networks to teach the CardSelector, we should not teach both networks at the same time, lest their learning interfers with itself.
'''
class CardSelector(object):

    def __init__(self, currentModelID = 0, cardfile = 'cards.txt'):
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(45, (1, 2), activation='relu', input_shape=(4, 4, 1), name = "input"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 2), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 2), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(.5))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.add(keras.layers.Dense(4, activation='softmax'))
        model.add(keras.layers.Reshape((1, 4)))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
        model.summary()

        self.model = model

    '''
    We attempt to have the selector network pick a better card to match the hand that
    '''
    def get_better_card_for_hand(self, hand, i, allcards, training = False):
        input = [0] * 16

        increment = 0
        for k in range(len(hand)):

            if increment != i:
                input[4 * increment + 0] = hand[k].left / 10
                input[4 * increment + 1] = hand[k].up / 10
                input[4 * increment + 2] = hand[k].right / 10
                input[4 * increment + 3] = hand[k].down / 10
                increment += 1



        output = self.model.predict(np.reshape(np.array(input, order='F', ndmin = 4), (-1, 4, 4, 1)))

        cardValues = np.reshape(output, (4)).tolist()

        if training:
            for i in range(len(cardValues)):
                cardValues[i] += (random.random() * 0.025 - 0.0125)
            randomCardMag = 0
            for n in cardValues:
                randomCardMag += n * n
            randomCardMag = math.sqrt(randomCardMag)
            for i in range(len(cardValues)):
                cardValues[i] /= randomCardMag
        cardDifferences = [(0, None)] * len(allcards)

        #find the difference in the l1 metric between our guess and card ratios, which to be honest should be precalculated, but aren't currently.
        #we opt out of the square root of the l2 metric at the end a quick optimization
        for k in range(len(allcards)):
            cardTotalValue = allcards[k].left + allcards[k].up + allcards[k].right + allcards[k].down

            leftNormalized = allcards[k].left / cardTotalValue
            upNormalized = allcards[k].up / cardTotalValue
            rightNormalized = allcards[k].right / cardTotalValue
            downNormalized = allcards[k].down / cardTotalValue

            leftDiffSqu = (cardValues[0] - leftNormalized)
            upDiffSqu = (cardValues[1] - upNormalized)
            rightDiffSqu = (cardValues[2] - rightNormalized)
            downDiffSqu = (cardValues[3] - downNormalized)

            cardDifferences[k] = (leftDiffSqu + upDiffSqu + rightDiffSqu + downDiffSqu, allcards[k])

        cardDifferences.sort(key=sort_tuple)

        for tuple in cardDifferences:
            #find first usable/valid card in the sort and return it.
            diff, newCard = tuple
            if (newCard.stars < 3):
                continue
            if (hand[i].stars <= 3 and newCard.stars >= 4): #was original card 4* or 5*? if not, only 3* or lower can work
                continue

            if (hand[i].stars >= 4 and newCard.stars < 4): #and if it was 4* or 5*, make it choose a 4* or 5*
                continue
            #make sure it is not a duplicate of one of the other four cards, but also skip the check on the ith card in the hand
            isDuplicate = False
            for card in hand:
                if card.name == newCard.name:
                    isDuplicate = True
                    break
            if isDuplicate:
                continue


            return newCard
        return None

class CardPredictor(object):
    def __init__(self, currentModelID = 0, cardfile = 'cards.txt'):
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', input_shape=(5, 29, 1), name = "input", padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(.5))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(15, activation='softmax'))
        model.add(keras.layers.Dense(8, activation='softmax'))
        model.add(keras.layers.Reshape((1, 8)))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
        model.summary()

        self.model = model

    def predictHiddenCards(self, gameboard, allcards, normalcards, legendarycards, cardnamedict, isP1, training = False):
        numCardsHidden = 2

        input = gameboard.getGameboardInputArray(allcards, normalcards, legendarycards, cardnamedict, isP1 = isP1)

        output = self.model.predict(np.reshape(np.array(input, order='F', ndmin = 4), (-1, 5, 29, 1)))



        cardValues = np.reshape(output, (8)).tolist()

        if training:
            for i in range(len(cardValues)):
                cardValues[i] += (random.random() * 0.2 - 0.1)

        hiddenCards = [None] * numCardsHidden

        for i in range(numCardsHidden):
            #find the difference in the l2 metric between our guess and card ratios, which to be honest should be precalculated, but aren't currently.
            #we opt out of the square root of the l2 metric at the end a quick optimization
            cardDifferences = [(0, None)] * len(allcards)

            cardTotal = math.sqrt(cardValues[4 * i + 0] * cardValues[4 * i + 0] + cardValues[4 * i + 1] * cardValues[4 * i + 1] + cardValues[4 * i + 2] * cardValues[4 * i + 2] + cardValues[4 * i + 3] * cardValues[4 * i + 3])


            for k in range(len(allcards)):
                cardTotalValue = allcards[k].left + allcards[k].up + allcards[k].right + allcards[k].down

                leftNormalized = allcards[k].left / cardTotalValue
                upNormalized = allcards[k].up / cardTotalValue
                rightNormalized = allcards[k].right / cardTotalValue
                downNormalized = allcards[k].down / cardTotalValue


                leftDiffSqu = (cardValues[4 * i + 0] / cardTotal - leftNormalized) * (cardValues[4 * i + 0] / cardTotal - leftNormalized)
                upDiffSqu = (cardValues[4 * i + 1] / cardTotal - upNormalized) * (cardValues[4 * i + 1] / cardTotal - upNormalized)
                rightDiffSqu = (cardValues[4 * i + 2] / cardTotal - rightNormalized) * (cardValues[4 * i + 2] / cardTotal - rightNormalized)
                downDiffSqu = (cardValues[4 * i + 3] / cardTotal - downNormalized) * (cardValues[4 * i + 3] / cardTotal - downNormalized)

                cardDifferences[k] = (leftDiffSqu + upDiffSqu + rightDiffSqu + downDiffSqu, allcards[k])

            cardDifferences.sort(key=sort_tuple)

            key, card = cardDifferences[0]
            hiddenCards[i] = card


        return hiddenCards

class ChoiceMaker(object):
    def __init__(self, currentModelID = 0, cardfile = 'cards.txt'):
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', input_shape=(5, 29, 1), name = "input", padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(45, (1, 15), activation='relu', padding = "same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(.5))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(45, activation='softmax'))
        model.add(keras.layers.Reshape((1, 45)))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
        model.summary()

        self.model = model
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
        otherInt = random.randint(0, 4)
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
        clone.p2Hand = self.initialP2Hand.copy()
        clone.p1Hand = self.p1Hand.copy()
        clone.p2Hand = self.p2Hand.copy()
        clone.p1Hidden = self.p1Hidden.copy()
        clone.p2Hidden = self.p2Hidden.copy()

        for i in range (9):
            if (self.gameboard[i] is not None):
                clone.gameboard[i] = self.gameboard[i].clone()
        clone.turn = self.turn
        clone.score = self.score
        return clone


    def flipHiddenCards(self, hand, isP1, allcards, normalcards, legendarycards, cardnamedict, hiddenCardPredictions = None):
        hand = hand.copy()
        #make a list of all hidden indexes and replace all hidden cards with our wildcard
        hiddenIndexes = []
        if self.turn % 2 == 0:
            if isP1:
                for i in range(len(hand)):
                    if hand[i] in self.initialP1Hand and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2): print(hiddenIndexes)

                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])


            else:
                for i in range(len(hand)):
                    if hand[i] in self.initialP1Hand and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2): print(hiddenIndexes)

                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])

        else:
            if isP1:
                for i in range(len(hand)):
                    if hand[i] in self.initialP2Hand and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2): print(hiddenIndexes)
                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])

            else:
                for i in range(len(hand)):
                    if hand[i] in self.initialP2Hand and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2): print(hiddenIndexes)
                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])


        return hand

    '''
    Yes this is actually necessary. Will thing of a better name / implementation later.
    '''
    def flipHiddenCards2(self, hand, isP1, allcards, normalcards, legendarycards, cardnamedict, hiddenCardPredictions = None):

        #make a list of all hidden indexes and replace all hidden cards with our wildcard
        hiddenIndexes = []
        if self.turn % 2 == 0:
            if isP1:
                for i in range(len(hand)):
                    if hand[i] in self.initialP2Hand and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2):
                            print(hiddenIndexes)
                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])

            else:
                for i in range(len(hand)):
                    if hand[i] in self.initialP1Hand and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2): print(hiddenIndexes)

                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])
        else:
            if isP1:
                for i in range(len(hand)):
                    if hand[i] in self.initialP1Hand and self.p1Hidden[self.initialP1Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2): print(hiddenIndexes)

                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])

            else:
                for i in range(len(hand)):
                    if hand[i] in self.initialP2Hand and self.p2Hidden[self.initialP2Hand.index(hand[i])]:
                        if (len(hiddenIndexes) >= 2): print(hiddenIndexes)

                        if hiddenCardPredictions is not None:
                            hand[i] = hiddenCardPredictions[len(hiddenIndexes)]
                            hiddenIndexes.append(i)
                        else:
                            hand[i] = Card(["Three Star Wild", 3, 8, 8, 8, 8])

        return hand

    def getGameboardInputArray(self, allcards, normalcards, legendarycards, cardnamedict, isP1 = False, hideCards = True, hiddenCardPredictions = None):
        redhand = self.p1Hand.copy() #TODO: dont actually call them this! fix at some point
        bluehand = self.p2Hand.copy()

        if hideCards:
            if (self.turn % 2 == 0):
                if isP1:
                    bluehand = self.flipHiddenCards2(bluehand, isP1, allcards, normalcards, legendarycards, cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
                else:
                    redhand = self.flipHiddenCards2(redhand, isP1, allcards, normalcards, legendarycards, cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
            else:
                if isP1:
                    redhand = self.flipHiddenCards2(redhand, isP1, allcards, normalcards, legendarycards, cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
                else:
                    bluehand = self.flipHiddenCards2(bluehand, isP1, allcards, normalcards, legendarycards, cardnamedict, hiddenCardPredictions = hiddenCardPredictions)
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
    Adapated from a friend's (Chie Tsukiya/NOR) code for an all open card solver

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
            if card in self.p1Hand:
                self.p1Hand.remove(card)
        else:
            #print("You can see your opponent has: (1) {}, (2) {}, (3) {}, (4) {}, (5) {}".format(self.p2Hand[0].name, self.p2Hand[1].name, self.p2Hand[2].name, self.p2Hand[3].name, self.p2Hand[4].name))
            if card in self.p2Hand:
                self.p2Hand.remove(card)
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
            if data[0] > 0:
                self.wins += 1
            elif data[1] > 0:
                self.wins += 0.5
        else:
            if data[2] > 0:
                self.wins += 1
            elif data[1] > 0:
                self.wins += 0.5
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
            if data[0] > 0:
                self.wins += 1
            elif data[1] > 0:
                self.wins += 0.5
        else:
            if data[2] > 0:
                self.wins += 1
            elif data[1] > 0:
                self.wins += 0.5
        self.total += 1
        for i in range(len(self.children)):
            if len(self.children) == 0:
                pass
            elif len(history) == 0:
                pass
            elif self.children[i].choice == history[0]:

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
        return totalDead >= turnsToDead(self.turn)
