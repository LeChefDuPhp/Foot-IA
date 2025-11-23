import sys
import pygame
from game import SoccerGameAI
from agent import train
import numpy as np
import torch
from model import ExtremeDuelingQNet
import config
import torch
from model import ExtremeDuelingQNet
import config
import argparse
import sys
from agent import Agent # Need Agent for get_opponent_state helper

def play():
    game = SoccerGameAI(render_mode=True)
    
    # Load model if exists
    # Input size is now 23 (14 + 8 + 1 comm)
    # Output size is 24 (6 actions * 4 comms)
    from model import ActorCritic
    model = ActorCritic(23, config.HIDDEN_LAYERS, config.ACTION_SIZE)
    model_loaded = False
    try:
        model.load_state_dict(torch.load('./model/model.pth', map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded! Playing against AI.")
        model_loaded = True
    except:
        print("No model found, playing against heuristic bot.")
        
    while True:
        # Human Control (Blue Player)
        keys = pygame.key.get_pressed()
        action_human = [0,0,0,0,0,0]
        
        if keys[pygame.K_UP]:
            action_human[1] = 1
        elif keys[pygame.K_DOWN]:
            action_human[2] = 1
        elif keys[pygame.K_LEFT]:
            action_human[3] = 1
        elif keys[pygame.K_RIGHT]:
            action_human[4] = 1
        elif keys[pygame.K_SPACE]:
            action_human[5] = 1
        else:
            action_human[0] = 1
            
        # AI Control (Red Player)
        action_bot = None
        if model_loaded:
            # Get state
            # Get Action for the AI (Red Player)
            # The model is trained from the perspective of Player 1 (left side).
            # When controlling the Red Player (right side), we need to feed it the opponent's state.
            state = game.get_state() # This state is from Blue Player's perspective
            
            # Get opponent's state (Red Player's perspective)
            # The get_opponent_state function expects a batch dimension, so reshape.
            opp_state_np = get_opponent_state(state.reshape(1, -1)).flatten()
            opp_state_tensor = torch.tensor(opp_state_np, dtype=torch.float).unsqueeze(0).to(model.device)
            
            # PPO returns prob, value. We just want action.
            action_probs, _ = model(opp_state_tensor)
            action_idx = torch.argmax(action_probs).item()
            
            # Convert to one-hot for game (if needed) or pass index
            # Game expects index or one-hot? _decode_action handles both.
            # Let's pass index to be safe as per agent.py
            action_bot = action_idx
        
        reward, done, score = game.play_step(action_human, action_bot)
        
        if done:
            game.reset()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer AI Game")
    parser.add_argument('mode', type=str, nargs='?', default='play',
                        help="Mode to run: 'train' or 'play'")
    args = parser.parse_args()

    if args.mode == 'train':
        import dashboard
        print("Starting Dashboard on http://localhost:8000")
        dashboard.start_dashboard_thread()
        train()
    else: # Default to play mode if no valid mode is specified or 'play' is chosen
        print("Usage: python main.py [train|play]")
        print("Defaulting to play mode...")
        play()
