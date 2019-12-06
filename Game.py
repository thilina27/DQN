import retro
import numpy as np
import skimage


class Game:
    def __init__(self, game_name):
        self.env = retro.make(game=game_name, players=1)
        self.actions_n = 8  # gym retro input
        self.life_count = 2
        self.image_height = 88
        self.image_width = 80
        self.number_of_frames = 4

    def reset(self):
        observation = self.env.reset()
        self.life_count = 2  # rest life
        # skip 60 frames
        for i in range(0, 64):
            self.step(0, reset=True)
        return self.step(0, reset=True)

    # action is a integer 0 - 3
    # 0 up 1 down 2 left 3 right
    # action = [0 ,0,0 ,0 ,up ,down ,left ,right]

    # do same action for 4 frames and collect all at once
    def do_action(self, action_space):
        # variables to keep data
        size = (self.image_height, self.image_width, self.number_of_frames)
        # create list to keep frames
        stack = np.zeros(size)
        reward = 0

        death = False
        terminal = False
        score = False
        for i in range(self.number_of_frames):

            state_, rew, done, info = self.env.step(action_space)
            # state data
            state_gray = self.preprocess_observation(state_)
            stack[:, :, i] = state_gray

            # change rewards based on death
            #if info['lives'] != self.life_count:
                #death = True

            # score reward
            if rew != 0:
                score = True

            # terminal in 4 frames
            if done:
                terminal = True

        #if death:
           # self.life_count -= 1
           # reward = -100
        if score:
            reward = 10
        else:
            reward = 0

        stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2])

        return stack, reward, terminal

    def step(self, action, reset=False):
        action_space = np.zeros(self.actions_n)
        if not reset:
            action_space[action] = 1
        return self.do_action(action_space)

    def render(self):
        self.env.render()

    # preprocessing fn - presenting pixels as signed bytes from -128 to 127 to have better replay memory (less costly)
    def preprocess_observation(self, obs):
        mspacman_colour = 210 + 164 + 74
        img = obs[1: 176:2, ::2] # downsize/crop images
        img = img.mean(axis=2) # turn to greyscale
        img[img == mspacman_colour] = 0 # constrast up
        return img.reshape(self.image_height, self.image_width) # reshape to 88 x 80 pixel