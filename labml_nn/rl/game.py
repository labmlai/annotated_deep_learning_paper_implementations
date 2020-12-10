"""
---
title: Atari wrapper with multi-processing
summary: This implements the Atari games with multi-processing.
---

# Atari wrapper with multi-processing
"""
import multiprocessing
import multiprocessing.connection

import cv2
import gym
import numpy as np


class Game:
    """
    ## <a name="game-environment"></a>Game environment
    This is a wrapper for OpenAI gym game environment.
    We do a few things here:

    1. Apply the same action on four frames and get the last frame
    2. Convert observation frames to gray and scale it to (84, 84)
    3. Stack four frames of the last four actions
    4. Add episode information (total reward for the entire episode) for monitoring
    5. Restrict an episode to a single life (game has 5 lives, we reset after every single life)

    #### Observation format
    Observation is tensor of size (4, 84, 84). It is four frames
    (images of the game screen) stacked on first axis.
    i.e, each channel is a frame.
    """

    def __init__(self, seed: int):
        # create environment
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.env.seed(seed)

        # tensor for a stack of 4 frames
        self.obs_4 = np.zeros((4, 84, 84))

        # buffer to keep the maximum of last 2 frames
        self.obs_2_max = np.zeros((2, 84, 84))

        # keep track of the episode rewards
        self.rewards = []
        # and number of lives left
        self.lives = 0

    def step(self, action):
        """
        ### Step
        Executes `action` for 4 time steps and
         returns a tuple of (observation, reward, done, episode_info).

        * `observation`: stacked 4 frames (this frame and frames for last 3 actions)
        * `reward`: total reward while the action was executed
        * `done`: whether the episode finished (a life lost)
        * `episode_info`: episode information if completed
        """

        reward = 0.
        done = None

        # run for 4 steps
        for i in range(4):
            # execute the action in the OpenAI Gym environment
            obs, r, done, info = self.env.step(action)

            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(obs)

            reward += r

            # get number of lives left
            lives = self.env.unwrapped.ale.lives()
            # reset if a life is lost
            if lives < self.lives:
                done = True
                break

        # maintain rewards for each step
        self.rewards.append(reward)

        if done:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None

            # get the max of last two frames
            obs = self.obs_2_max.max(axis=0)

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward, done, episode_info

    def reset(self):
        """
        ### Reset environment
        Clean up episode info and 4 frame stack
        """

        # reset OpenAI Gym environment
        obs = self.env.reset()

        # reset caches
        obs = self._process_obs(obs)
        for i in range(4):
            self.obs_4[i] = obs
        self.rewards = []

        self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4

    @staticmethod
    def _process_obs(obs):
        """
        #### Process game frames
        Convert game frames to gray and rescale to 84x84
        """
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    """
    ##Worker Process

    Each worker process runs this method
    """

    # create game
    game = Game(seed)

    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    """
    Creates a new worker and runs it in a separate process.
    """

    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()


