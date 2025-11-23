import multiprocessing as mp
from game import SoccerGameAI
import numpy as np

def worker(remote, parent_remote):
    parent_remote.close()
    game = SoccerGameAI(render_mode=False)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action_ai, action_bot, action_ai_2, action_bot_2 = data
            reward, done, score = game.play_step(action_ai, action_bot, action_ai_2, action_bot_2)
            if done:
                game.reset(level=game.level)
            state = game.get_state()
            remote.send((state, reward, done, score))
        elif cmd == 'reset':
            level = data
            game.reset(level=level)
            state = game.get_state()
            remote.send(state)
        elif cmd == 'close':
            remote.close()
            break

class ParallelEnv:
    def __init__(self, n_envs=16):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote))
                   for work_remote, remote in zip(self.work_remotes, self.remotes)]
        
        for p in self.ps:
            p.daemon = True
            p.start()
            
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions_ai, actions_bot, actions_ai_2=None, actions_bot_2=None):
        if actions_ai_2 is None: actions_ai_2 = [None] * self.n_envs
        if actions_bot_2 is None: actions_bot_2 = [None] * self.n_envs
        
        for remote, action_ai, action_bot, action_ai_2, action_bot_2 in zip(self.remotes, actions_ai, actions_bot, actions_ai_2, actions_bot_2):
            remote.send(('step', (action_ai, action_bot, action_ai_2, action_bot_2)))
        results = [remote.recv() for remote in self.remotes]
        states, rewards, dones, scores = zip(*results)
        return np.stack(states), np.stack(rewards), np.stack(dones), scores

    def reset(self, level=3):
        for remote in self.remotes:
            remote.send(('reset', level))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
