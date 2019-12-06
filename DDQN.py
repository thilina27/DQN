import random
from collections import deque

from keras import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
from Game import Game

IMAGE_HEIGHT = 88
IMAGE_WIDTH = 80
# model parameters
LEARNING_RATE = 0.001
STACK_SIZE = 4  # stack size for single state
BATCH = 50  # size of mini batch
GAMMA = 0.99
ACTIONS_N = 8

# agent
FINAL_EPSILON = 0.1  # final value of epsilon
INITIAL_EPSILON = 1  # starting value of epsilon
OBSERVER = 60  # filling D (experience replay data)
REPLAY_SIZE = 50000  # size of D

#
EPS_DECAY_SIZE = 850000
TOTAL_EPI = 900000
C = 1000  # update q`

ENV_NAME = 'MsPacman-Atari2600'


# -- Brain -- #
class Brain:

    def __init__(self):
        # DQN
        self.model = self._create_model()   # q model
        self._model = self._create_model()  # q` model ( used to calculate predication for error)
        self.training_loss = 0
        self.batch_id = 0

    # no init of network param : we can use normal dist.-:init=lambda shape, name: normal(shape, scale=0.01, name=name),
    def _create_model(self):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, STACK_SIZE)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(ACTIONS_N))
        model.add(Activation('linear'))

        #opt = RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=0.01)
        opt = Adam(lr=LEARNING_RATE)

        def huber_loss(y, q_value):
            error = K.abs(y - q_value)
            quadratic_part = K.clip(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
            return loss

        model.compile(loss=huber_loss, optimizer=opt)

        return model

     # predict action from state images
    def predict_action(self, state):

        q = self.model.predict(state)[0]  # input a stack of 4 images, get the prediction
        print("Action Q : ", q)
        max_q = np.argmax(q)
        action_val = max_q

        return action_val

    # train model using the re play queue
    def train(self, mini_batch):

        self.training_loss = 0

        inputs = np.zeros((BATCH, IMAGE_HEIGHT, IMAGE_WIDTH, STACK_SIZE))  # 2500, 100, 100, 4
        targets = np.zeros((inputs.shape[0], ACTIONS_N))  # 2500, 4
        targets_ = np.zeros((inputs.shape[0], ACTIONS_N))  # 2500, 4

        # Now we do the experience replay
        for j in range(0, len(mini_batch)):
            state_t = mini_batch[j][0]  # state
            action_t = mini_batch[j][1]  # action
            reward_t = mini_batch[j][2]  # reward
            state_t1 = mini_batch[j][3]  # new state
            terminal = mini_batch[j][4]  # is terminal reached or not

            inputs[j:j + 1] = state_t  # saved down s_t as input

            # predict q values for current state
            targets[j] = self.model.predict(state_t)[0]
            # predict q values for next state
            targets_[j] = self.model.predict(state_t1)[0]
            # print("model prediction st    : ", targets[j])
            # print("model prediction st1   : ", targets_[j])

            q_sa = self._model.predict(state_t1)[0]  # predict to get arg max Q to cal TD
            # print("_model prediction st1  : ", q_sa)

            if terminal:
                targets[j, action_t] = reward_t  # if terminal only set target as reward for the action
            else:
                targets[j, action_t] = reward_t + GAMMA * q_sa[np.argmax(targets_[j])]

        logs = self.model.train_on_batch(inputs, targets)
        print("loss : ", logs) 

    # update q`
    def update_target_model(self):
        self._model = self.model

    def save(self, model_name):
        self.model.save(model_name)


# -- Agent -- #
class Agent:

    def __init__(self):
        self.D = deque()
        self.epsilon = INITIAL_EPSILON
        self.brain = Brain()
        self.modelCount = 0

    # do action
    def act(self, state):

        if random.random() <= self.epsilon:
            self.brain.predict_action(state)
            action_val = self.act_random()

        else:
            action_val = self.brain.predict_action(state)  # input a stack of 4 images, get the prediction

        return action_val

    def act_random(self):
        return random.randrange(ACTIONS_N)

    def observe(self, state, action_value, reward, new_state, terminal_reached):

        if len(self.D) > REPLAY_SIZE:
            self.D.popleft()
        self.D.append((state, action_value, reward, new_state, terminal_reached))

    def replay(self):
        # sample a mini batch to train on
        mini_batch = random.sample(self.D, BATCH)
        self.brain.train(mini_batch)

    def update_brain(self):
        self.brain.update_target_model()

    def update_epsilon(self):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / REPLAY_SIZE

    def save_brain(self):
        model_name = 'testModel' + str(self.modelCount) + '.h5'
        self.brain.save(model_name)
        self.modelCount += 1


# -- env -- #

class Environment:

    def __init__(self):
        self.agent = Agent()
        self.game = Game(ENV_NAME)

    def run(self):
        #  fill D by random
        count = 0
        for i in range(0, OBSERVER):
            state_t, _, _ = self.game.reset()
            #self.game.render()
            terminal = False
            while not terminal:
                action = self.agent.act_random()
                state_t1, reward, terminal = self.game.step(action)
                self.agent.observe(state_t, action, reward, state_t1, terminal)
                #print(terminal)
                state_t = state_t1
                count += 1
                #self.game.render()
            if i % 10 == 0:
                print(i, count)

        # train agent

        for i in range(0, TOTAL_EPI):
            terminal = False
            state_t, _, _ = self.game.reset()
            #print(state_t.shape)
            rr = 0
            frames = 0
            while not terminal:
                frames += 1
                action = self.agent.act(state_t)
                state_t1, reward, terminal = self.game.step(action)
                self.agent.observe(state_t, action, reward, state_t1, terminal)
                # train every fourth frame
                if frames % 4 == 0:
                    self.agent.replay()
                state_t = state_t1
                rr += reward
            self.agent.update_epsilon()
            print("Episode ", i, " Reward : ", rr)

            if i != 0 and i % C == 0:
                self.agent.update_brain()
            if i % 100 == 0:
                self.save_model()
                print(i)
                total_rew = 0
                for j in range(0, 10):
                    terminal = False
                    state_t, _, _ = self.game.reset()
                    self.game.render()
                    while not terminal:
                        action = self.agent.act(state_t)
                        state_t1, reward, terminal = self.game.step(action)
                        total_rew += reward
                        state_t = state_t1
                        self.game.render()
                print("Step : ", i, " Mean Reward : ", total_rew/10)

    def test(self, model_name):
        self.game = Game(ENV_NAME)
        # todo : fill method to test saved models
        # load model based on the model name : Agent -> brain
        # select action from loaded model brain : Agent
        # perform action to game
        # keep rewards (accumulate to termination)

    def save_model(self):
        self.agent.save_brain()


# -- Main -- #

environment = Environment()

try:
    print("start")
    environment.run()
finally:
    environment.save_model()
