#-----Sonic 'Info' Accessible Values
#'act': 0, 
#'screen_x': 0, 
#'zone': 0, 
#'level_end_bonus': 0, 
#'screen_y': 768, 
#'score': 0, 
#'lives': 3, 
#'screen_x_end': 9407, 
#'rings': 0, 
#'x': 55, 
#'y': 945


#---Action Space Labels
#Action 0: Jump
#Action 1: Jump
#Action 2: ???
#Action 3: ???
#Action 4: Up
#Action 5: Down
#Action 6: Left
#Action 7: Right
#Action 8: Jump
#Action 9: ???
#Action 10: ???
#Action 11: ???
#Action 12: ???


#NetOutput
# 0: Jump
# 1: Up
# 2: Down
# 3: Left
# 4: Right
#===========================================================
import cv2          # pip install opencv-python
import numpy as np
from math import log, sqrt
#===========================================================
def sonicrun(brain, env, render=True):
    obs = env.reset()
    obs, rew, done, info = env.step(env.action_space.sample())
    act = info['act']
    obs = env.reset()
    inx = int(obs.shape[0]/8)
    iny = int(obs.shape[1]/8)

    xpos_max = 0
    cycles = 0
    lastchange = 0
    jumpcnt = 0
    stoptime = 250

    while True:
        cycles += 1
        if render:
            env.render()
        if cycles-lastchange < 250.0:
            obs = cv2.resize(obs, (inx, iny))
#            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (1, inx, iny, 3))
            netoutput = brain.predict(obs)[0]
            netoutput = [round(x) for x in netoutput]

            controllist = np.array([0 for i in range(12)])
            controllist[0] = netoutput[0]
            if controllist[0] == 1:
                jumpcnt += 1
            controllist[4] = netoutput[1]
            controllist[5] = netoutput[2]
            controllist[6] = netoutput[3]
            controllist[7] = netoutput[4]
        else:
            obs, rew, done, info = env.step(env.action_space.sample())



        try:
            obs, rew, done, info = env.step(controllist)
        except TypeError:
            print(controllist)
            raise TypeError

        xpos = info['x']
        if xpos > xpos_max:
            xpos_max = xpos
            lastchange = cycles
        lives = info['lives']
        if lives != 3:
            done = True
        curact = info['act']
        if act != curact:
            done = True

        if cycles-lastchange > 250.0+1.75*xpos_max or done:
#        if done:
            break


    distfromgoal = float(9407-xpos)

    if distfromgoal < 20.0:
        finishreward = -3000
    else:
        finishreward = 0
        
    finaltime = float(cycles)
    objective = 100.0*sqrt(cycles) + 50.0*distfromgoal + 0.5*jumpcnt + finishreward

    return objective


if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten
    import retro
    import sys
    env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")    
    actionsize = len(env.action_space.sample())
    obs = env.reset()
    inx = int(obs.shape[0]/8)
    iny = int(obs.shape[1]/8)
    print(type(env.action_space.sample()))
    env.reset()

    model = Sequential()
    model.add(Conv2D(102, kernel_size=5, activation='tanh', input_shape=(inx,iny,3)))
    model.add(Conv2D(50, kernel_size=3, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    try:
        filename = sys.argv[1]
    except IndexError:
        raise IndexError("Please Pass in the network file as an argument.")
    model.load_weights(filename)
    score = sonicrun(model, env, render=True)
    print("Objective Score: %s"%(score))

