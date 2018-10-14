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

        # Initialize the belief to be 1 at the initial position for each of the
        # opposition agents. The predictions will be a dictionary of dictionaries.
        # The inner dictionaries will hold the predictions for each agent.
        self.predictions = {}
        for enemy in self.enemies:
            self.predictions[enemy] = util.Counter()
            self.predictions[enemy][gameState.getInitialAgentPosition(enemy)] = 1.


    def initialisePredictions(self, enemy):
        """
        Initializing a uniform distribution for the predictions. Meaning that when we have no knowledge
        of the state, we can assume that it is equally likely that the agent
        could be in any pos.
        """

        self.predictions[enemy] = util.Counter()

        for pos in self.availablePos:
            # This value of 1, could be anything since we will normalize it.
            initial = 1.0
            self.predictions[enemy][pos] = initial

        self.predictions[enemy].normalize()


    def sleepTime(self, enemy, gameState):
        """
        In this case, we will set the distribution by looking at all the
        possible successor positions and checking that they are legal positions.
        Of the legal positions we will set it to be uniformly likely to
        transition to the legal state.
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

    # If the the true distance to the position is less than or equal to
            # 5 then we know this cannot be the true position because we
            # would have gotten it as a true reading then and not a noisy
            # distance reading so we can set the belief to 0.        
            if manDistance <= 5:
                foundPrediction[pos] = 0.
            elif pac != gameState.getAgentState(enemy).isPacman:
                foundPrediction[pos] = 0.
            else:
                # This equation is the online belief update that is given by
                # the following equation:
                # P(x_t|e_1:t) = P(x_t|e_1:t) * P(e_t:x_t).
                foundPrediction[pos] = self.predictions[enemy][pos] * emissionModel

        if foundPrediction.totalCount() == 0:
            self.initialisePredictions(enemy)
            # Normalize and set the new belief.y)
        else:
            foundPrediction.normalize()
            self.predictions[enemy] = foundPrediction


    def chooseAction(self, gameState):
        """
        Base choose action. In this function we begin by updating our predictions
        and elapsing time for the predictions. We also show our predictions on the
        screen by using the provided debugging function.
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

        #  self.displayDistributionsOverPositions(self.predictions.values())

        # Using the most probable position update the game state.
        # In order to use expectimax we need to be able to have a set
        # position where the enemy is starting out.
        for enemy in self.enemies:
            probablePosition = self.predictions[enemy].argMax()
            conf = game.Configuration(probablePosition, Directions.STOP)
            newState.data.agentStates[enemy] = game.AgentState(conf, newState.isRed(probablePosition) != newState.isOnRedTeam(enemy))

        # Do expectimax to depth 2 and get the best action to use. This is
        # the furthest out that we could do this because of time constraints.
        action = self.getEnemyMoves(newState, depth=1)[1]
        
        if len(self.getFood(gameState).asList()) <= 0:
            print ("You have to go back.")
        return action


    def getEnemyMoves(self, gameState, depth):
        """
        This is the getEnemyMoves of expectimax in HW2. We are are choosing the
        move to maximize our expected utility for the agent on our team.
        This is done by also using the getExpectimaxValue from HW2 to get
        the expected result of the enemy moves.
        """

        # Check for the end of the game or reaching the end of the recursion.
        if depth == 0 or gameState.isOver():
            return self.eval(gameState), Directions.STOP

        # Get the successor game states for the possible moves.
        nextActions = gameState.getLegalActions(self.index)

        # We found better results when we always required a move.
        nextActions.remove(Directions.STOP)
        successorgamestates = [gameState.generateSuccessor(self.index, action)
                                 for action in nextActions]

        
        
        # Get the expected allscores of enemy moves.
        allscores = [self.getExpectimaxValue(successorGameState, self.enemies[0], depth)[0]
                    for successorGameState in successorgamestates]

        if len(successorgamestates) > 4:
            print ("allscores-",allscores)
            print ("successorgamestate -",successorgamestates)
            print ("enemies-",self.enemies)
            print ("len allscores-",len(allscores))
        
        scoreBest = max(allscores)
        BestIndexes = [index for index in range(len(allscores)) if
                         allscores[index] == scoreBest]
        chosenIndex = random.choice(BestIndexes)

        return scoreBest, nextActions[chosenIndex]


    def getExpectimaxValue(self, gameState,enemy,depth):   
        """
        This is the expectimax function from HW2. This will be called for
        each of the enemy agents. Once it goes to the next level we will use
        the max function again since we will be back on our team.
        """
        # Check for end of game or reaching end of the recursion.te, enemy, depth):
    

        
        if depth == 0 or gameState.isOver():
            return self.eval(gameState), Directions.STOP

        # Get the successor game states for the possible moves.
        enemyactions = gameState.getLegalActions(enemy)
        successorgamestates = []
        for action in enemyactions:
            try:
                successorgamestates.append(gameState.generateSuccessor(enemy, action))
            except:
                pass

        # If there is another ghost, then call the expecti function for the
        # next ghost, otherwise call the max function for pacman.
        scores = [self.getEnemyMoves(successorGameState, depth - 1)[0]
                        for successorGameState in successorgamestates]

        # Calculate the expected value.
        bestValue = sum(scores) / len(scores)
        
        return bestValue, Directions.STOP


    def enemyDist(self, gameState):
        """
        If we are getting a reading for the agent distance then we will return
        this exact distance. In the case that the agent is beyond our sight
        range we will assume that the agent is in the position where our
        belief is the highest and return that position. We will then get the
        distances from the agent to the enemy.
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
    """
    An offensive agent that will immediately head for the side of the opposing
    team and will never chase agents on its own team side. We use several
    features and weights that we iterated to improve by viewing games and
    results. The agent also has limits on carrying so that it will go back
    to the other side after collecting a number of food.
    """

    def registerInitialState(self, gameState):
        parentAgent.registerInitialState(self, gameState)
        self.retreating = False


    def chooseAction(self, gameState):
        score = self.getScore(gameState)
        horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
        # Choose how many food to collect before attempting to turn back based
        # off the score of the game.(self.initialFood/2)-2
        
        if score < 8:
            eatFood = 6
        else:
        # Do not set as a retreating agent if the carrying limit is not reached
        # or there is the minimum amount of food left.      
            eatFood = 4

       
        if gameState.getAgentState(self.index).numCarrying < eatFood and len(self.getFood(gameState).asList()) > 2:
            self.retreating = False
        else:
            if min(horrorCount) > 4: # Do not retreat but se5rch for food.
                self.retreating = False
            else:
                self.retreating = True

        return parentAgent.chooseAction(self, gameState)


    def eval(self, gameState):
        # Get the current position.
        currentPosition = gameState.getAgentPosition(self.index)

        # Get the food on the board.
        nextFood = self.getFood(gameState).asList()

        # Get the closest distance to the middle of the board.
        distanceCovered = min([self.distancer.getDistance(currentPosition, (self.widthDivision, i))
                                 for i in range(gameState.data.layout.height)
                                 if (self.widthDivision, i) in self.availablePos])

        # Getting the distances to the enemy agents that are ghosts.
        ghostDist = []
        for enemy in self.enemies:
            if not gameState.getAgentState(enemy).isPacman:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    ghostDist.append(self.distancer.getDistance(currentPosition, enemyPos))
        # Get the minimum distance of any of the ghost distances.
        # If it is greater than 4, we do not care about it so make it 0

        minGhostDistances = min(ghostDist) if len(ghostDist) else 0
        if minGhostDistances >= 4:
            minGhostDistances = 0

        # Get whether there is a power pill we are chasing.
        foodChasing = None
        if self.red:
            foodChasing = gameState.getBlueCapsules()
        else:
            foodChasing = gameState.getRedCapsules()

        # distance and minimum distance to the capsule.
        foodChasingDist = [self.distancer.getDistance(currentPosition, capsule) for capsule in
                                    foodChasing]
        minFoodChasingDist = min(foodChasingDist) if len(foodChasingDist) else 0

        # Time to go back to safety, or trying to find food still.
        if self.retreating:
            # Want to get back to the other side at this point. Weight is on
            # staying safe and getting back to the halfway point.
            return - 2 * distanceCovered + 500 * minGhostDistances
        else:
            # Actively looking for food.
            foodDist = [self.distancer.getDistance(currentPosition, food) for
                             food in nextFood]
            minFoodDist = min(foodDist) if len(foodDist) else 0
            horrorCount = [gameState.getAgentState(enemy).scaredTimer for enemy
                             in self.enemies]

            # If they are scared be aggressive.
            if min(horrorCount) <= 6 and minGhostDistances < 4:
                minGhostDistances *= -1

            return 2 * self.getScore(gameState) - 100 * len(nextFood) - \
                   3 * minFoodDist - 10000 * len(foodChasing) - \
                   5 * minFoodChasingDist + 100 * minGhostDistances


class DefensiveAgent(parentAgent):
    """
    This is a defensive agent that likes to attack. If there are no enemy pacman
    then the defensive agent will act on the offensive agent evaluation function.
    We do not use carry limits though because the agent will retreat when the
    other team has a pacman.
    """
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

        # If there are no pacman on our side or the poison pill is active we
        # should act like an offensive agent.
        if (numPacmans == 0 or min(horrorCount) > 8) and len(self.getFood(gameState).asList()) > 2:
            self.offensing = True
        else:
            self.offensing = False

        return parentAgent.chooseAction(self, gameState)


    def eval(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)

        # Get the most likely enemy distances.
        enemyDist = self.enemyDist(gameState)

        # Get the pacman on our side.
        checkPacman = [pacEnemies for pacEnemies in self.enemies if
                    gameState.getAgentState(pacEnemies).isPacman]

        # Get the distance to the pacman and find the minimum.
        pacdist = [dist for id, dist in enemyDist if
                         gameState.getAgentState(id).isPacman]
        minPacDist = min(pacdist) if len(pacdist) else 0

        # Find the distance to the ghosts and find the minimum.
        distToGhost = [dist for id, dist in enemyDist if
                             not gameState.getAgentState(id).isPacman]
        minGhostDist = min(distToGhost) if len(distToGhost) else 0

        # Get min distance to pacEnemies food.
        enemyFood = self.getFood(gameState).asList()
        foodDist = [self.distancer.getDistance(currentPosition, food) for food in
                         enemyFood]
        minFoodDist = min(foodDist) if len(foodDist) else 0

        # Get min distance to pacEnemies power pill.
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

