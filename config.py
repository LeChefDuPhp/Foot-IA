import torch

# Game Settings
WIDTH = 800
HEIGHT = 600
FPS = 60

# Physics Constants
FRICTION = 0.98
ACCELERATION = 1.5
MAX_SPEED = 10
BALL_FRICTION = 0.99
BALL_MAX_SPEED = 15
KICK_FORCE = 12
AIR_DRAG = 0.995
MAGNUS_STRENGTH = 0.1
SPIN_DECAY = 0.99

# AI Hyperparameters
BATCH_SIZE = 8192 # Massive batch size for RX 7800 XT
LR = 0.0003
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
PPO_EPOCHS = 10
MAX_MEMORY = 1_000_000
HIDDEN_LAYERS = [2048, 1024, 512, 256] # Deep network for complex strategy

# Communication
COMM_SIZE = 4 # Number of discrete messages (0, 1, 2, 3)
PHYSICAL_ACTIONS = 6 # Idle, Up, Down, Left, Right, Kick
ACTION_SIZE = PHYSICAL_ACTIONS * COMM_SIZE # 24 Combined actions

# Training Settings
PARALLEL_ENVS = 16
OPPONENT_UPDATE_GAMES = 100
OPPONENT_WINRATE_THRESHOLD = 0.60
CHECKPOINT_DIR = './checkpoints'
MODEL_DIR = './model'

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Curriculum Learning
CURRICULUM_LEVELS = {
    1: "SHOOTING",   # Ball near goal, No Bot
    2: "DRIBBLING",  # Ball in center, No Bot
    3: "DUEL",       # Full 1v1
    4: "2v2"         # 2v2 Mode
}
CURRICULUM_THRESHOLD = 0.80 # Win rate to advance

# 2v2 Settings
TEAM_SIZE = 2 # Players per team
STATE_SIZE_2V2 = 26 # Expanded state vector for 4 players + ball


