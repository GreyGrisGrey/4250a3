import random

# Our function for Vhat(s, w) is a linear function w[0]*x[0] w[1]*x[1]. For the affine function the constant is zero
def Vhat(curr, w):
    return (curr[0] * w[0] + curr[1] * w[1])

def move(x, y, action):
    next = [x+action[0], y+action[1]]
    if next[0] > 6 or next[1] > 6 or next[1] < 0 or next[0] < 0:
        return ([x, y], 0)
    elif next[0] == 0 and next[1] == 0:
        return ([0, 0], -1)
    elif next[0] == 6 and next[1] == 6:
        return ([6, 6], 1)
    return ([x+action[0], y+action[1]], 0)

def TD():
    # No need to open a text file this time since we aren't going to use it anyway, just set up a blank set of states
    # Policy is equiprobable moves and no value function is created, so we save a lot of space and time compared to other questions
    states = []
    actions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    for i in range(7):
        stateRow = []
        for j in range(7):
            stateRow.append(".")
        states.append(stateRow)
    # The gridworld is very simple, having the gradient vector be simple too seemed most reasonable
    # Feature vectors encode X position and Y position
    gradient = [0, 0]
    alpha = 1
    gamma = 0.5
    # TD(0) only considers the current state and the next state at any given calculation.
    # So it seemed reasonable to start it in a random location each time step.
    # It also selects a random action at each time step
    curr = None
    for i in range(1000000):
        if curr == None:
            curr = (3, 3)
        action = actions[random.randint(0, 3)]
        next, val = move(curr[0], curr[1], action)
        Ut = val + gamma * Vhat((curr[0]+action[0], curr[1]+action[1]), gradient)
        gradientMod = alpha * (Ut - Vhat((curr[0], curr[1]), gradient))
        gradient[0] += gradientMod * curr[0]
        gradient[1] += gradientMod * curr[1]
        alpha = 1/((i+1)/50)
        curr = next
    # Prints the gradient vector, as above gradient[0] is the x value and gradient[1] is the y value
    print(gradient)

def MonteCarlo():
    # No need to open a text file this time since we aren't going to use it anyway, just set up a blank set of states
    # Policy is equiprobable moves and no value function is created, so we save a lot of space and time compared to other questions
    states = []
    actions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    for i in range(7):
        stateRow = []
        for j in range(7):
            stateRow.append(".")
        states.append(stateRow)
    # The gridworld is very simple, having the gradient vector be simple too seemed most reasonable
    # Feature vectors encode X position and Y position
    gradient = [0, 0]
    alpha = 0.1
    gamma = 0.9
    # Gradient Monte Carlo requires us to maintain a list of visited states and actions typically
    # This is to compute the sequence of rewards acquired over the trajectory of the actor
    # Here it suffices to store a list of actions and just go backwards
    roundCount = 0
    curr = None
    actionList = []
    for i in range(100000):
        if curr == None:
            curr = [3, 3]
            start = [3, 3]
        action = actions[random.randint(0, 3)]
        curr, reward = move(curr[0], curr[1], action)
        actionList.append(action)
        # Rewards don't need to be stored in this case as the only time they would not be 0 is upon hitting a terminal state
        if reward != 0:
            roundCount += 1
            for i in range(len(actionList)):
                gradientMod = alpha * (reward*(gamma**i) - Vhat((curr[0], curr[1]), gradient))
                gradient[0] += gradientMod * curr[0]
                gradient[1] += gradientMod * curr[1]
                if (curr[0] - actionList[len(actionList)-(i+1)][0]) <= 6 and (curr[0] - actionList[len(actionList)-(i+1)][0]) >= 0:
                    curr[0] -= actionList[len(actionList)-(i+1)][0]
                if (curr[1] - actionList[len(actionList)-(i+1)][1]) <= 6 and (curr[1] - actionList[len(actionList)-(i+1)][1]) >= 0:
                    curr[1] -= actionList[len(actionList)-(i+1)][1]
            alpha = 0.05/(roundCount)
            actionList = []
            curr = None
    # Prints the gradient vector, as above gradient[0] is the x value and gradient[1] is the y value
    print(gradient)

TD()
MonteCarlo()