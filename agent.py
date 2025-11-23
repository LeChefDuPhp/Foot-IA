import torch
import random
import numpy as np
from collections import deque
from game import SoccerGameAI, Direction
from model import ActorCritic, PPOTrainer
from helper import plot
from parallel_env import ParallelEnv
import config
import os
import shared_state
import time

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = config.GAMMA
        self.memory = deque(maxlen=config.MAX_MEMORY)
        
        # PPO Memory
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        # Main Model (Player 1)
        # Input size 23 to accommodate 2v2 state + comm
        self.model = ActorCritic(23, config.HIDDEN_LAYERS, config.ACTION_SIZE)
        self.trainer = PPOTrainer(
            self.model, 
            lr=config.LR, 
            gamma=config.GAMMA, 
            gae_lambda=config.GAE_LAMBDA,
            clip_epsilon=config.CLIP_EPSILON,
            entropy_coef=config.ENTROPY_COEF,
            value_loss_coef=config.VALUE_LOSS_COEF,
            epochs=config.PPO_EPOCHS
        )
        
        # Opponent Model (Player 2 - Frozen)
        self.opponent_model = ActorCritic(23, config.HIDDEN_LAYERS, config.ACTION_SIZE)
        self.opponent_model.load_state_dict(self.model.state_dict()) # Copy weights
        self.opponent_model.eval() # Inference only
        self.opponent_model.to(self.trainer.device)
        
        self.env = ParallelEnv(n_envs=config.PARALLEL_ENVS)
        
        # Stats for Self-Play Update
        self.wins = 0
        self.games_in_batch = 0
        self.generation = 0
        self.level = 1 # Start at Level 1
        
        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)

    def save_checkpoint(self):
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"manual_save_gen_{self.generation}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Manual Checkpoint saved: {checkpoint_path}")
        self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        try:
            checkpoints = [os.path.join(config.CHECKPOINT_DIR, f) for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pth')]
            checkpoints.sort(key=os.path.getmtime)
            
            while len(checkpoints) > config.MAX_CHECKPOINTS:
                oldest_checkpoint = checkpoints.pop(0)
                os.remove(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")
        except Exception as e:
            print(f"Error rotating checkpoints: {e}")

    def get_action(self, state, model=None):
        if model is None: model = self.model
        
        state = torch.tensor(state, dtype=torch.float).to(self.trainer.device)
        
        with torch.no_grad():
            action_probs, value = model(state)
            
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def get_opponent_state(self, state):
        # Flip state for opponent perspective
        # State: [AI_x, AI_y, AI_vx, AI_vy, Bot_x, Bot_y, Bot_vx, Bot_vy, Ball_x, Ball_y, Ball_vx, Ball_vy, ...]
        # Flip X coordinates (assuming 0-1 normalized)
        # x -> 1-x, vx -> -vx
        
        opp_state = np.array(state, copy=True)
        
        # Indices for 1v1 (0-13)
        # AI (0-3) <-> Bot (4-7)
        # Swap AI and Bot
        opp_state[:, [0,1,2,3, 4,5,6,7]] = opp_state[:, [4,5,6,7, 0,1,2,3]]
        
        # Flip X (0, 4, 8)
        opp_state[:, 0] = 1.0 - opp_state[:, 0] # AI x
        opp_state[:, 4] = 1.0 - opp_state[:, 4] # Bot x
        opp_state[:, 8] = 1.0 - opp_state[:, 8] # Ball x
        
        # Flip Vx (2, 6, 10)
        opp_state[:, 2] = -opp_state[:, 2]
        opp_state[:, 6] = -opp_state[:, 6]
        opp_state[:, 10] = -opp_state[:, 10]
        
        # 2v2 Flips (Indices 14-21)
        # AI_Team (14-17) <-> Bot_Team (18-21)
        if opp_state.shape[1] > 14:
             opp_state[:, [14,15,16,17, 18,19,20,21]] = opp_state[:, [18,19,20,21, 14,15,16,17]]
             # Flip X
             opp_state[:, 14] = 1.0 - opp_state[:, 14]
             opp_state[:, 18] = 1.0 - opp_state[:, 18]
             # Flip Vx
             opp_state[:, 16] = -opp_state[:, 16]
             opp_state[:, 20] = -opp_state[:, 20]
             
        return opp_state

    def compute_gae(self, next_value, done_mask):
        values = self.vals + [next_value]
        rewards = self.rewards
        dones = self.dones
        
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + config.GAMMA * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
            
        return returns

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    
    # Initial Reset
    states = agent.env.reset(level=agent.level)
    
    while True:
        # Check Controls
        controls = shared_state.state.get_controls()
        if controls["paused"]:
            time.sleep(1)
            continue
            
        if controls["save_requested"]:
            agent.save_checkpoint()
            shared_state.state.set_control("save_requested", False)
            
        # Update LR
        for param_group in agent.trainer.optimizer.param_groups:
            param_group['lr'] = controls["learning_rate"]

        # 1. Get Actions (PPO: Sample)
        actions_idx, log_probs, values = agent.get_action(states)
        
        # Convert indices to one-hot for game (if game expects one-hot) or just pass indices
        # Game expects one-hot or index? Let's check game.py. 
        # It handles index in _decode_action now.
        
        # 2. Get Opponent Actions
        opp_states = agent.get_opponent_state(states)
        opp_actions_idx, _, _ = agent.get_action(opp_states, agent.opponent_model)
        
        # 3. Step
        # We need to pass actions for 4 players if 2v2
        # For now, assuming 1v1 or 2v2 handled by ParallelEnv unpacking
        # ParallelEnv step expects lists.
        
        # If 2v2, we need actions for teammates.
        # Current Agent controls P1.
        # Who controls P2 (Teammate)? 
        # Ideally, same agent controls both (Shared Weights).
        # So we generate actions for P2 using same model.
        
        actions_ai_2 = None
        actions_bot_2 = None
        
        if agent.level == 4:
            # Generate actions for teammates
            # Teammate sees state from their perspective? 
            # For simplicity, let's assume teammate sees same state but is in different position?
            # No, we need proper state for teammate.
            # For now, let's just use the SAME state (Hive Mind) or random?
            # BETTER: Use same model, same state (Hive Mind).
            actions_ai_2_idx, _, _ = agent.get_action(states)
            actions_bot_2_idx, _, _ = agent.get_action(opp_states, agent.opponent_model)
            actions_ai_2 = actions_ai_2_idx
            actions_bot_2 = actions_bot_2_idx

        next_states, rewards, dones, scores = agent.env.step(actions_idx, opp_actions_idx, actions_ai_2, actions_bot_2)
        
        # 4. Store Memory
        # Flatten parallel data? No, keep as lists, then flatten before train.
        # Actually PPO usually collects T steps for N envs.
        # We store (s, a, log_p, r, d, v)
        agent.states.append(states)
        agent.actions.append(actions_idx)
        agent.probs.append(log_probs)
        agent.vals.append(values)
        agent.rewards.append(rewards)
        agent.dones.append(dones)
        
        states = next_states
        
        # Update Stats
        agent.n_games += sum(dones)
        agent.wins += sum([1 for r in rewards if r > 5]) # Rough win check
        agent.games_in_batch += sum(dones)
        
        # 5. Train if Batch Full
        if len(agent.states) >= config.BATCH_SIZE // config.PARALLEL_ENVS:
            # Compute GAE
            # Need value of NEXT state
            _, _, next_values = agent.get_action(next_states)
            
            # Flatten buffers
            # (T, N, D) -> (T*N, D)
            
            # Compute returns for each env
            # This is tricky with parallel envs in flat list.
            # We need to transpose: (T, N) -> (N, T)
            
            T = len(agent.rewards)
            N = config.PARALLEL_ENVS
            
            # Convert to numpy (T, N)
            np_rewards = np.array(agent.rewards)
            np_dones = np.array(agent.dones)
            np_vals = np.array(agent.vals)
            np_next_vals = next_values
            
            # Compute GAE per env
            batch_returns = []
            batch_advantages = []
            
            for i in range(N):
                env_rewards = np_rewards[:, i]
                env_dones = np_dones[:, i]
                env_vals = np_vals[:, i]
                env_next_val = np_next_vals[i]
                
                # GAE
                returns = []
                gae = 0
                for t in reversed(range(T)):
                    next_val = env_vals[t+1] if t < T-1 else env_next_val
                    delta = env_rewards[t] + config.GAMMA * next_val * (1 - env_dones[t]) - env_vals[t]
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - env_dones[t]) * gae
                    returns.insert(0, gae + env_vals[t])
                
                batch_returns.append(returns)
                batch_advantages.append(np.array(returns) - env_vals)
                
            # Flatten everything
            flat_states = np.array(agent.states).reshape(-1, 23)
            flat_actions = np.array(agent.actions).reshape(-1)
            flat_probs = np.array(agent.probs).reshape(-1)
            flat_returns = np.array(batch_returns).T.reshape(-1)
            flat_advantages = np.array(batch_advantages).T.reshape(-1)
            
            loss = agent.trainer.train_step(flat_states, flat_actions, flat_probs, flat_returns, flat_advantages)
            
            agent.clear_memory()
            agent.generation += 1
            
            # Save Checkpoint
            if agent.generation % 10 == 0:
                agent.save_checkpoint()
                
            # Update Shared State
            shared_state.state.update_stats(
                loss=loss,
                generation=agent.generation,
                level=agent.level,
                games_played=agent.n_games
            )

        # Plotting & Opponent Update (Same as before)
        if sum(dones) > 0:
            avg_score = np.mean(scores) # This is cumulative score? 
            # ParallelEnv returns current score.
            # We want average score of finished games.
            # For simplicity, just plot current avg score of all envs
            
            plot_mean_scores.append(avg_score)
            plot(plot_scores, plot_mean_scores)
            
            shared_state.state.update_stats(
                mean_score=avg_score,
                win_rate=agent.wins / (agent.games_in_batch if agent.games_in_batch > 0 else 1)
            )
            
        if agent.games_in_batch >= config.OPPONENT_UPDATE_GAMES:
            win_rate = agent.wins / agent.games_in_batch
            if win_rate > config.OPPONENT_WINRATE_THRESHOLD:
                print(">>> UPGRADING OPPONENT <<<")
                agent.opponent_model.load_state_dict(agent.model.state_dict())
                
                # Save Checkpoint
                agent.generation += 1
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"gen_{agent.generation}.pth")
                torch.save(agent.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                agent._rotate_checkpoints()
                
                # Curriculum
                if win_rate > config.CURRICULUM_THRESHOLD and agent.level < 4:
                    print(f">>> PROMOTED TO LEVEL {agent.level + 1} ({config.CURRICULUM_LEVELS[agent.level + 1]}) <<<")
                    agent.level += 1
                    agent.env.reset(level=agent.level)
            
            agent.wins = 0
            agent.games_in_batch = 0
        
        states = next_states

if __name__ == '__main__':
    train()
