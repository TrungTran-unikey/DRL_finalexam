from random import randint
import numpy as np
import gymnasium as gym

DECODE_ACTION = {
    0: 'left',
    1: 'right',
    2: 'up',
    3: 'down',
    4: 'stop',
}

class ChaserEvader(gym.Env):
    def __init__(self, Chaser, Evader, speed=0.5):
        self.Chaser = Chaser
        self.Evader = Evader
        self.speed = speed
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(5,))
        self.state = None
        self.done = False
        self.reward = 0
        self.info = {}
        self.steps = 0
        self.max_steps = 1000
        self.reset()
        
    def reset(self):
        self.Evader.x = randint(-9, 9)
        self.Evader.z = randint(-9, 9)

        self.Chaser.x = randint(-9, 9)
        self.Chaser.z = randint(-9, 9)

        while self._distance(self.Chaser, self.Evader) <= 2:
            self.Chaser.x = randint(-9, 9)
            self.Chaser.z = randint(-9, 9)

        distance = self._distance(self.Chaser, self.Evader)

        self.done = False

        self.reward = {
            'Chaser': 0,
            'Evader': 0,
        }

        self.steps = 0
        self.info = {}
        self.state = {
            'Chaser': [self.Chaser.x, self.Chaser.z, distance, self.Evader.x, self.Evader.z,],
            'Evader': [self.Evader.x, self.Evader.z, distance, self.Chaser.x, self.Chaser.z,],
        }
        self.time = 20
        self.prev_distance = distance
        return self.state, {}
    
    def step(self, action:dict):
        self.steps += 1
        Chaser_action = action['Chaser']
        Evader_action = action['Evader']

        self.move(self.Chaser, DECODE_ACTION[Chaser_action], self.speed)
        self.move(self.Evader, DECODE_ACTION[Evader_action], self.speed)

        distance = self._distance(self.Chaser, self.Evader)
        
        self.state = {
            'Chaser': [self.Chaser.x, self.Chaser.z, distance, self.Evader.x, self.Evader.z,],
            'Evader': [self.Evader.x, self.Evader.z, distance, self.Chaser.x, self.Chaser.z,],
        }

        self.done = self.if_done()

        self.reward = {
            'Chaser': self._reward_Chaser(distance),
            'Evader': self._reward_Evader(distance),
        }

        self.prev_distance = distance

        return self.state, self.reward, self.done, self.info
    
    def move(self, player, action, speed:float):
        if action == 'left':
            player.x -= speed
        elif action == 'right':
            player.x += speed
        elif action == 'up':
            player.z += speed
        elif action == 'down':
            player.z -= speed
        elif action == 'stop':
            player.x = player.x
            player.z = player.z
        else:
            print('Invalid action')

        self._check_wall_collision(player)
    
    def _check_wall_collision(self, player):
        if player.x >= 9:
            player.x = 8.5
        elif player.x <= -9:
            player.x = -8.5
        elif player.z >= 9:
            player.z = 8.5
        elif player.z <= -9:
            player.z = -8.5

    def _distance(self, player1, player2):
        return np.sqrt((player1.x - player2.x)**2 + (player1.z - player2.z)**2)
    
    def _reward_Evader(self, distance):
        if self.time <= 1.5e-2:
            return 1
        
        elif self.done and self.time > 1.5e-2:
            return -1
        
        return 0.01
    
    def _reward_Chaser(self, distance):
        if self.time <= 1.5e-2:
            return -1
        
        elif self.done and self.time > 1.5e-2:
            return 1
        
        if round(distance) < round(self.prev_distance):
            return 0.01

        return -0.01

    def if_done(self):
        if self.Evader.intersects(self.Chaser) or self.time <= 1.5e-2:
            return True
        
        return False

    def render(self):
        pass