# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions
import game
import distanceCalculator
from util import nearestPoint
from math import floor

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed, first='OffensiveAgent',
                 second='DefensiveAgent'):
    """
    This function returns a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class parentAgent(CaptureAgent):
    
    def registerInitialState(self, gameState):
        
        self.initialFood = len(self.getFood(gameState).asList())
        
        CaptureAgent.registerInitialState(self, gameState)
        self.heightDivision = gameState.data.layout.height/2
        self.availablePos = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        self.start = gameState.getInitialAgentPosition(self.index)
        self.widthDivision = gameState.data.layout.width/2
        self.offensing = False
        self.distancer.getMazeDistances()

        self.team = self.getTeam(gameState)

  

        self.enemies = self.getOpponents(gameState)
        self.predictions = {}
        for enemy in self.enemies:
            self.predictions[enemy] = util.Counter()
            self.predictions[enemy][gameState.getInitialAgentPosition(enemy)] = 1.


    def initialisePredictions(self, enemy):
        """
        Initializing a uniform distribution for the predictions.
        """

        self.predictions[enemy] = util.Counter()

        for pos in self.availablePos:
            initial = 1.0
            self.predictions[enemy][pos] = initial

        self.predictions[enemy].normalize()


    def sleepTime(self, enemy, gameState):
        """
        Function to calculate the elapse time
        """
        newPredict = util.Counter()

        for everyPos in self.availablePos:
           
            possiblePositions = [(everyPos[0]+i, everyPos[1]+j) for i in [-1,0,1]
                                 for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]

            newDist = util.Counter()
            for p in possiblePositions:
                if p in self.availablePos:
                    newDist[p] = 1.0
                else:
                    pass

           
            newDist.normalize()

          
            for newPos, prob in newDist.items():
                
                newPredict[newPos] += prob * self.predictions[enemy][everyPos]
        newPredict.normalize()
        self.predictions[enemy] = newPredict


    def trackEnemy(self, enemy, observation, gameState):
        
        """
        Function to track the enemy agent positions keeping in track the noisy distances
        """

       
        noisyDistance = observation[enemy]

        currentPos = gameState.getAgentPosition(self.index)

        foundPrediction = util.Counter()

     
        for pos in self.availablePos:
           
            manDistance = util.manhattanDistance(currentPos, pos)

            emissionModel = gameState.getDistanceProb(manDistance, noisyDistance)

            if self.red:
                pac = pos[0] < self.widthDivision
            else:
                pac = pos[0] > self.widthDivision      
            if manDistance <= 5:
                foundPrediction[pos] = 0.
            elif pac != gameState.getAgentState(enemy).isPacman:
                foundPrediction[pos] = 0.
            else:
                # prediction or belief update formula
                foundPrediction[pos] = self.predictions[enemy][pos] * emissionModel

        if foundPrediction.totalCount() == 0:
            self.initialisePredictions(enemy)
        else:
            foundPrediction.normalize()
            self.predictions[enemy] = foundPrediction


    def chooseAction(self, gameState):
        """
        This function is used updating our predictions and elapsing time based on the predictions.
        """
        

        currentPos = gameState.getAgentPosition(self.index)
        noisyDistances = gameState.getAgentDistances()
        newState = gameState.deepCopy()

        for enemy in self.enemies:
            enemyPos = gameState.getAgentPosition(enemy)
            if enemyPos:
                predict = util.Counter()
                predict[enemyPos] = 1.0
                self.predictions[enemy] = predict
            else:
                self.sleepTime(enemy, gameState)
                self.trackEnemy(enemy, noisyDistances, gameState)

   
        for enemy in self.enemies:
            probablePosition = self.predictions[enemy].argMax()
            conf = game.Configuration(probablePosition, Directions.STOP)
            newState.data.agentStates[enemy] = game.AgentState(conf, newState.isRed(probablePosition) != newState.isOnRedTeam(enemy))


        action = self.getEnemyMoves(newState, depth=2)[1]

        return action


    def getEnemyMoves(self, gameState, depth):
        """
        This is the getEnemyMoves of expectimax.
        """

        # End of game when depth is 0 
        if depth == 0 or gameState.isOver():
            return self.eval(gameState), Directions.STOP

        # next actions of successirs
        nextActions = gameState.getLegalActions(self.index)

        nextActions.remove(Directions.STOP)
        successorgamestates = [gameState.generateSuccessor(self.index, action)
                                 for action in nextActions]

        allscores = [self.getExpectimaxValue(successorGameState, self.enemies[0], depth)[0]
                    for successorGameState in successorgamestates]

        scoreBest = max(allscores)
        BestIndexes = [index for index in range(len(allscores)) if
                         allscores[index] == scoreBest]
        chosenIndex = random.choice(BestIndexes)

        return scoreBest, nextActions[chosenIndex]


    def getExpectimaxValue(self, gameState,enemy,depth):   
    

        if depth == 0 or gameState.isOver():
            return self.eval(gameState), Directions.STOP

        enemyactions = gameState.getLegalActions(enemy)
        successorgamestates = []
        for action in enemyactions:
            try:
                successorgamestates.append(gameState.generateSuccessor(enemy, action))
            except:
                pass

        if enemy < max(self.enemies):
            scores = [self.getExpectimaxValue(successorGameState, enemy + 2, depth)[0]
                        for successorGameState in successorgamestates]
        else:
            scores = [self.getEnemyMoves(successorGameState, depth - 1)[0]
                        for successorGameState in successorgamestates]

        # The expected value
        bestValue = sum(scores) / len(scores)

        return bestValue, Directions.STOP


    def enemyDist(self, gameState):
        """
        Used for tracking enemy distances
        """
        distances = []
        for enemy in self.enemies:
            currentPosition = gameState.getAgentPosition(self.index)
            enemyLoc = gameState.getAgentPosition(enemy)
            if enemyLoc:  # This is the case we know the exact position.
                pass
            else:  # If we don't know exact position, get most likely.
                enemyLoc = self.predictions[enemy].argMax()
            distances.append((enemy, self.distancer.getDistance(currentPosition, enemyLoc)))
        return distances


    def eval(self, gameState):
        """
        Evaluate the utility of a game state.
        """
        util.raiseNotDefined()


class OffensiveAgent(parentAgent):
 

    def registerInitialState(self, gameState):
        parentAgent.registerInitialState(self, gameState)
        self.retreating = False


    def chooseAction(self, gameState):
        score = self.getScore(gameState)
        horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
        
        if score < floor((self.initialFood/2))-2:
            valOne = floor((self.initialFood/3))
            eatFood = valOne
        else:  
            valTwo = floor((self.initialFood/4))   
            eatFood = valTwo

       
        if gameState.getAgentState(self.index).numCarrying < eatFood and len(self.getFood(gameState).asList()) > 2:
            self.retreating = False
        else:
            if min(horrorCount) > 4: 
                self.retreating = False
            else:
                self.retreating = True

        return parentAgent.chooseAction(self, gameState)


    def eval(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)

        nextFood = self.getFood(gameState).asList()
        distanceCovered = min([self.distancer.getDistance(currentPosition, (self.widthDivision, i))
                                 for i in range(gameState.data.layout.height)
                                 if (self.widthDivision, i) in self.availablePos])

        ghostDist = []
        for enemy in self.enemies:
            if not gameState.getAgentState(enemy).isPacman:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    ghostDist.append(self.distancer.getDistance(currentPosition, enemyPos))

        if len(ghostDist):
            minGhostDistances = min(ghostDist)
        else :
            minGhostDistances = 0
            
        if minGhostDistances >= 4:
            minGhostDistances = 0
            
        foodChasing = None
        if self.red:
            foodChasing = gameState.getBlueCapsules()
        else:
            foodChasing = gameState.getRedCapsules()

        foodChasingDist = [self.distancer.getDistance(currentPosition, capsule) for capsule in
                                    foodChasing]
        if len(foodChasingDist):
            minFoodChasingDist = min(foodChasingDist)
        else :
            minFoodChasingDist = 0

        if self.retreating:
     
            return - 2 * distanceCovered + 500 * minGhostDistances
        else:
            
            foodDist = [self.distancer.getDistance(currentPosition, food) for
                             food in nextFood]
            if len(foodDist): 
                minFoodDist = min(foodDist) 
            else :
                minFoodDist = 0
            horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy
                             in self.enemies]

            if min(horrorCount) <= 6 and minGhostDistances < 4:
                minGhostDistances *= -1

            return 2 * self.getScore(gameState) - 100 * len(nextFood) - \
                   3 * minFoodDist - 10000 * len(foodChasing) - \
                   5 * minFoodChasingDist + 100 * minGhostDistances


class DefensiveAgent(parentAgent):
    
    def registerInitialState(self, gameState):
        parentAgent.registerInitialState(self, gameState)
        self.offensing = False


    def chooseAction(self, gameState):
        # Check if the enemy has any pacman.
        checkPacman = [pacEnemies for pacEnemies in self.enemies if
                    gameState.getAgentState(pacEnemies).isPacman]
        numPacmans = len(checkPacman)

        # Check if we have the poison active.
        horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy in
                         self.enemies]

        # If there are no pacmans then act as an offensive agent
        if numPacmans == 0 or min(horrorCount) > 8:
            self.offensing = True
        else:
            self.offensing = False

        return parentAgent.chooseAction(self, gameState)


    def eval(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)

    
        enemyDist = self.enemyDist(gameState)

        checkPacman = [pacEnemies for pacEnemies in self.enemies if
                    gameState.getAgentState(pacEnemies).isPacman]

        # Get the distance to the pacman and find the minimum.
        pacdist = [dist for id, dist in enemyDist if
                         gameState.getAgentState(id).isPacman]
                
        if len(pacdist):
            minPacDist = min(pacdist)
        else :
            minPacDist = 0 

        distToGhost = [dist for id, dist in enemyDist if
                             not gameState.getAgentState(id).isPacman]
        if len(distToGhost) :
            minGhostDist = min(distToGhost)
        else :
            minGhostDist = 0
            
        enemyFood = self.getFood(gameState).asList()
        foodDist = [self.distancer.getDistance(currentPosition, food) for food in
                         enemyFood]
        if len(foodDist):
            minFoodDist = min(foodDist)
        else : 
            minFoodDist = 0

        powerpill = self.getCapsulesYouAreDefending(gameState)
        distToPowerPill = [self.getMazeDistance(currentPosition, capsule) for capsule in
                             powerpill]
        if len(distToPowerPill):
            minPillDist = min(distToPowerPill) 
        else :
            minPillDist = 0
        MAXVAL = 999999
        if self.offensing == False:
            return -MAXVAL * len(checkPacman) - 10 * minPacDist - minPillDist
        else:
            return 2 * self.getScore(gameState) - 100 * len(enemyFood) - \
                   3 * minFoodDist + minGhostDist

