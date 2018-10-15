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


    """
    This is the main agent will be that will contain the defintion of the child agents 

    """
    
    def registerInitialState(self, gameState):
        
        self.initialFood = len(self.getFood(gameState).asList())
        

        CaptureAgent.registerInitialState(self, gameState)

        #cordianted of the center of the grid will be sent the pacman agent to go the initial position 
        self.heightDivision = gameState.data.layout.height/2
        self.availablePos = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        self.start = gameState.getInitialAgentPosition(self.index)
        self.widthDivision = gameState.data.layout.width/2
        
        #this will set defensive agent into offensive mode 
        self.offensing = False
        self.distancer.getMazeDistances()


        self.team = self.getTeam(gameState)
        self.carryLimit = None
  

        #enemies will store the info of all the enemies
        self.enemies = self.getOpponents(gameState)

      
        self.predictions = {}
        for enemy in self.enemies:
            self.predictions[enemy] = util.Counter()
            self.predictions[enemy][gameState.getInitialAgentPosition(enemy)] = 1.



    def initialisePredictions(self, enemy):


        """
        This function will predict the initial value for the pacman

        """
        self.predictions[enemy] = util.Counter()

        for pos in self.availablePos:
            # This value of 1, could be anything since we will normalize it.
            initial = 1.0
            self.predictions[enemy][pos] = initial

        self.predictions[enemy].normalize()


    def sleepTime(self, enemy, gameState):
  
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
        function that will be used to track the enemy by using the 
        getOpponents position and calculating the distance between them

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
   
                foundPrediction[pos] = self.predictions[enemy][pos] * emissionModel

        if foundPrediction.totalCount() == 0:
            self.initialisePredictions(enemy)
        else:
            foundPrediction.normalize()
            self.predictions[enemy] = foundPrediction


    def chooseAction(self, gameState):

        """
        Main function that will decide the action to be take on the basis of
        Expitamax technique. 
        We are simulating the game to the depth of 2 levels 
        and rewards are caculated on the basis of those simulations

        """

        #currentPos will store the current position of the agent
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


        action = self.getEnemyMoves(newState, depth=1)[1]
        
        return action


    def getEnemyMoves(self, gameState, depth):


        """
        This function will predict the next move of the enemy at all dept levels
        and will return the probable score to the attacking agent to evaluate the 
        next step

        """

        if depth == 0 or gameState.isOver():
            return self.eval(gameState), Directions.STOP

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
        

        """

        This is main evaluation function used to caluclate the expected maximum value

        """
        if depth == 0 or gameState.isOver():
            return self.eval(gameState), Directions.STOP

        enemyactions = gameState.getLegalActions(enemy)
        successorgamestates = []
        for action in enemyactions:
            try:
                successorgamestates.append(gameState.generateSuccessor(enemy, action))
            except:
                pass


        #this will calculate the score for ever successor game state and its enemyies location
        scores = [self.getEnemyMoves(successorGameState, depth - 1)[0]
                        for successorGameState in successorgamestates]

        bestValue = sum(scores) / len(scores)
        
        return bestValue, Directions.STOP


    def enemyDist(self, gameState):

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

        util.raiseNotDefined()


class OffensiveAgent(parentAgent):

    """
    This agent will alawys be in attacking mode and will have a carry capacity of
    1/3rd of the pacdots

    """
    def registerInitialState(self, gameState):
        parentAgent.registerInitialState(self, gameState)
        self.retreating = False


    def chooseAction(self, gameState):

        """
        This is main funciton will that will return the action derived from the 
        Expitamax function.
        This funciton will also determine the carry limit of the agent.
        """
        score = self.getScore(gameState)
        horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]

        
        if score < 8:
            parentAgent.carryLimit = 4
        else:
   
            parentAgent.carryLimit = 2

       
        if gameState.getAgentState(self.index).numCarrying < parentAgent.carryLimit:
            self.retreating = False
        elif len(self.getFood(gameState).asList()) > 2:
            self.retreating = False
        else:
            if min(horrorCount) > 4: # Do not retreat but se5rch for food.
                self.retreating = False
            else:
                self.retreating = True

        return parentAgent.chooseAction(self, gameState)


    def eval(self, gameState):

        """
        This funciton will evaluate on the basis of weights which includes 
        minGhostDistances, foodChasingDist, scaredTimer and next food

        """

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

        #This will make sure pacman agent stays at a minimum distance of 4 from the ghost
        minGhostDistances = min(ghostDist) if len(ghostDist) else 0
        if minGhostDistances >= 4:
            minGhostDistances = 0


        foodChasing = None
        if self.red:
            foodChasing = gameState.getBlueCapsules()
        else:
            foodChasing = gameState.getRedCapsules()

        #This will store minimum distance for the next food in target
        foodChasingDist = [self.distancer.getDistance(currentPosition, capsule) for capsule in
                                    foodChasing]
        minFoodChasingDist = min(foodChasingDist) if len(foodChasingDist) else 0

   
        if self.retreating:

            return - 2 * distanceCovered + 500 * minGhostDistances
        else:

            foodDist = [self.distancer.getDistance(currentPosition, food) for
                             food in nextFood]
            minFoodDist = min(foodDist) if len(foodDist) else 0
            horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy
                             in self.enemies]

            if min(horrorCount) <= 6 and minGhostDistances < 4:
                minGhostDistances *= -1

            #this will return the most optimal score required to take the next step
            return 2 * self.getScore(gameState) - 100 * len(nextFood) - \
                   3 * minFoodDist - 10000 * len(foodChasing) - \
                   5 * minFoodChasingDist + 100 * minGhostDistances


class DefensiveAgent(parentAgent):

    """
    This agent will behave differently on various scenarios.
    1. It will be offensive if there is no pacman in our area
    2. It will be defensive if there is one or more agent in our area
    3. It will be defensive if there is less than 2 pacdost available
    """
 
    def registerInitialState(self, gameState):
        parentAgent.registerInitialState(self, gameState)
        self.offensing = False


    def chooseAction(self, gameState):
        """
        This function will determine the action for the defensive agent
        on the basis of enemy in vicinity and state of offensive agent
        """
        checkPacman = [pacEnemies for pacEnemies in self.enemies if
                    gameState.getAgentState(pacEnemies).isPacman]
        numPacmans = len(checkPacman)

        horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy in
                         self.enemies]

        if (numPacmans == 0 or min(horrorCount) > 8) and len(self.getFood(gameState).asList()) > 2:
            self.offensing = True
        else:
            self.offensing = False

        return parentAgent.chooseAction(self, gameState)


    def eval(self, gameState):

        """
        This funciton will evaluate on the basis of weights which includes 
        minGhostDistances, pacDist, scaredTimer and powerpill

        """
        currentPosition = gameState.getAgentPosition(self.index)

        enemyDist = self.enemyDist(gameState)

        checkPacman = [pacEnemies for pacEnemies in self.enemies if
                    gameState.getAgentState(pacEnemies).isPacman]

        pacdist = [dist for id, dist in enemyDist if
                         gameState.getAgentState(id).isPacman]
        minPacDist = min(pacdist) if len(pacdist) else 0

        distToGhost = [dist for id, dist in enemyDist if
                             not gameState.getAgentState(id).isPacman]
        minGhostDist = min(distToGhost) if len(distToGhost) else 0

        enemyFood = self.getFood(gameState).asList()
        foodDist = [self.distancer.getDistance(currentPosition, food) for food in
                         enemyFood]
        minFoodDist = min(foodDist) if len(foodDist) else 0

        powerpill = self.getCapsulesYouAreDefending(gameState)
        distToPowerPill = [self.getMazeDistance(currentPosition, capsule) for capsule in
                             powerpill]
        minPillDist = min(distToPowerPill) if len(distToPowerPill) else 0
        MAXVAL = 999999
        if self.offensing == False:
            return -MAXVAL * len(checkPacman) - 10 * minPacDist - minPillDist
        else:
            return 2 * self.getScore(gameState) - 100 * len(enemyFood) - \
                   3 * minFoodDist + minGhostDist

