import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import math

pygame.init()
font = pygame.font.SysFont('arial', 25)

import config

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Colors (Dark Mode / Cyberpunk)
BLACK = (10, 10, 15)
WHITE = (240, 240, 255)
RED = (255, 50, 50)
BLUE = (50, 50, 255)
NEON_GREEN = (57, 255, 20)
GRAY = (50, 50, 60)

BLOCK_SIZE = 20
SPEED = config.FPS

class SoccerGameAI:
    def __init__(self, w=config.WIDTH, h=config.HEIGHT, render_mode=True):
        self.w = w
        self.h = h
        self.render_mode = render_mode
        
        if self.render_mode:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Neural Football 1v1')
        
        self.clock = pygame.time.Clock()
        
        # Load Assets (Only in Render Mode)
        if self.render_mode:
            try:
                self.field_img = pygame.image.load('assets/field.png').convert()
                self.field_img = pygame.transform.scale(self.field_img, (self.w, self.h))
                
                self.player_img = pygame.image.load('assets/player.png').convert_alpha()
                self.player_img = pygame.transform.scale(self.player_img, (BLOCK_SIZE*2, BLOCK_SIZE*2))
                
                self.ball_img = pygame.image.load('assets/ball.png').convert_alpha()
                self.ball_img = pygame.transform.scale(self.ball_img, (int(BLOCK_SIZE), int(BLOCK_SIZE)))
                
                self.assets_loaded = True
            except Exception as e:
                print(f"Warning: Could not load assets: {e}")
                self.assets_loaded = False
                
            self.particles = [] # List of [x, y, vx, vy, life, color]

        self.reset()

    def reset(self, level=3):
        # Game State
        self.score_ai = 0
        self.score_bot = 0
        self.frame_iteration = 0
        self.level = level
        self.match_time = config.MATCH_DURATION * config.FPS # Frames
        
        self._reset_positions()
        return self.get_state()

    def _reset_positions(self):
        # Entities (x, y, vx, vy)
        self.ai_pos = np.array([self.w/4, self.h/2], dtype=float)
        self.ai_vel = np.array([0.0, 0.0], dtype=float)
        
        self.bot_pos = np.array([3*self.w/4, self.h/2], dtype=float)
        self.bot_vel = np.array([0.0, 0.0], dtype=float)
        
        self.ball_pos = np.array([self.w/2, self.h/2], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)
        self.ball_spin = 0.0
        
        # Curriculum Scenarios
        if self.level == 1: # SHOOTING
            # Ball random position in opponent half
            self.ball_pos[0] = random.randint(int(self.w/2), int(self.w - 50))
            self.ball_pos[1] = random.randint(50, int(self.h - 50))
            # Bot removed (far away)
            self.bot_pos = np.array([-1000.0, -1000.0])
            
        elif self.level == 2: # DRIBBLING
            # Ball in center
            self.ball_pos = np.array([self.w/2, self.h/2], dtype=float)
            # Bot removed
            self.bot_pos = np.array([-1000.0, -1000.0])
            
        # Level 3 is default (Duel)
        
        if self.level == 4: # 2v2
            # Team 1 (AI)
            self.ai_pos = np.array([self.w/4, self.h/3], dtype=float) # P1 Top
            self.ai_teammate_pos = np.array([self.w/4, 2*self.h/3], dtype=float) # P2 Bottom
            self.ai_teammate_vel = np.array([0.0, 0.0], dtype=float)
            
            # Team 2 (Bot)
            self.bot_pos = np.array([3*self.w/4, self.h/3], dtype=float) # P1 Top
            self.bot_teammate_pos = np.array([3*self.w/4, 2*self.h/3], dtype=float) # P2 Bottom
            self.bot_teammate_vel = np.array([0.0, 0.0], dtype=float)
            
            self.ball_pos = np.array([self.w/2, self.h/2], dtype=float)
        else:
            # Hide extra players
            self.ai_teammate_pos = np.array([-1000.0, -1000.0])
            self.ai_teammate_vel = np.array([0.0, 0.0])
            self.bot_teammate_pos = np.array([-1000.0, -1000.0])
            self.bot_teammate_vel = np.array([0.0, 0.0])
        
        # Communication State (Last message sent by each player)
        # 0 is default/silence
        self.comm_ai = 0
        self.comm_ai_teammate = 0
        self.comm_bot = 0
        self.comm_bot_teammate = 0
        
        self.comm_bot = 0
        self.comm_bot_teammate = 0
        
        self.last_dist_to_ball = np.linalg.norm(self.ai_pos - self.ball_pos)
        
        # Possession Tracking (0=None, 1=AI, 2=AI_Teammate, 3=Bot, 4=Bot_Teammate)
        self.last_touch_player = 0
        
    def _move_entity(self, pos, vel, action_vec=None):
        # Apply acceleration if action is taken
        if action_vec is not None:
            vel += action_vec * config.ACCELERATION
            
        # Apply friction
        vel *= config.FRICTION
        
        # Cap speed
        speed = np.linalg.norm(vel)
        if speed > config.MAX_SPEED:
            vel = (vel / speed) * config.MAX_SPEED
            
        # Update position
        pos += vel
        
        # Boundary checks (bounce off walls)
        if pos[0] < 0:
            pos[0] = 0
            vel[0] *= -0.5
        elif pos[0] > self.w:
            pos[0] = self.w
            vel[0] *= -0.5
            
        if pos[1] < 0:
            pos[1] = 0
            vel[1] *= -0.5
        elif pos[1] > self.h:
            pos[1] = self.h
            vel[1] *= -0.5
            
        return pos, vel

    def _move_ball(self):
        # 1. Apply Air Drag (Linear drag approximation)
        self.ball_vel *= config.AIR_DRAG
        
        # 2. Apply Magnus Effect (Force perpendicular to velocity)
        # F_m = S x V
        # 2D Cross product: (vx, vy) rotated 90 deg is (-vy, vx)
        # Force direction depends on spin sign.
        if np.linalg.norm(self.ball_vel) > 0.1:
            magnus_force = np.array([-self.ball_vel[1], self.ball_vel[0]]) * self.ball_spin * config.MAGNUS_STRENGTH
            self.ball_vel += magnus_force
            
        # 3. Apply Ground Friction
        self.ball_vel *= config.BALL_FRICTION
        
        # 4. Decay Spin
        self.ball_spin *= config.SPIN_DECAY
        
        speed = np.linalg.norm(self.ball_vel)
        if speed > config.BALL_MAX_SPEED:
            self.ball_vel = (self.ball_vel / speed) * config.BALL_MAX_SPEED
            
        self.ball_pos += self.ball_vel
        
        # Wall collisions (Bounce)
        if self.ball_pos[1] < 0 or self.ball_pos[1] > self.h:
            self.ball_vel[1] *= -0.9
            self.ball_pos[1] = np.clip(self.ball_pos[1], 0, self.h)
            # Wall hit reduces spin
            self.ball_spin *= 0.8
            
        # Goal Lines (x-axis)
        # If ball goes out on sides but NOT in goal, bounce? 
        # For simplicity: Goals are the entire width of the wall for now, or we define posts.
        # Let's define goals as the middle 1/3 of the wall.
        goal_top = self.h/3
        goal_bottom = 2*self.h/3
        
        if self.ball_pos[0] < 0:
            if goal_top < self.ball_pos[1] < goal_bottom:
                return 2 # Bot Scored (AI Conceded)
            else:
                self.ball_vel[0] *= -0.9
                self.ball_pos[0] = 0
                
        elif self.ball_pos[0] > self.w:
            if goal_top < self.ball_pos[1] < goal_bottom:
                return 1 # AI Scored
            else:
                self.ball_vel[0] *= -0.9
                self.ball_pos[0] = self.w
                
        return 0 # No goal

    def _kick_ball(self, player_pos, player_vel, action):
        p_radius = BLOCK_SIZE
        b_radius = BLOCK_SIZE/2

        direction = self.ball_pos - player_pos
        dist = np.linalg.norm(direction)
        if dist == 0: direction = np.array([1.0, 0.0]) # Default direction if at same spot
        else: direction = direction / dist
        
        # Calculate Spin based on "offset" of the kick
        rel_vel = self.ball_vel - player_vel
        tangent = np.array([-direction[1], direction[0]])
        spin_imparted = np.dot(rel_vel, tangent) * 0.5
        self.ball_spin += spin_imparted
        
        force = config.KICK_FORCE
        if action is not None and np.argmax(action) == 5: force *= 1.5 # Power shot
        
        self.ball_vel += direction * force

    def _check_collisions(self, action_ai, action_bot, action_ai_2=None, action_bot_2=None):
        reward = 0
        done = False
        score = 0
        
        p_radius = BLOCK_SIZE
        b_radius = BLOCK_SIZE/2
        
        # --- Ball Collisions ---
        
        # AI 1
        dist_ai = np.linalg.norm(self.ball_pos - self.ai_pos)
        if dist_ai < (p_radius + b_radius):
            self._kick_ball(self.ai_pos, self.ai_vel, action_ai)
            reward = 1 # Touch ball
            
            # Pass Reward (Receive)
            if self.last_touch_player == 2:
                reward += config.REWARD_PASS
                # We could also reward the passer (Teammate) here if we had access to their reward buffer
                
            self.last_touch_player = 1
            
        # AI 2 (Teammate)
        if self.level == 4:
            dist_ai_2 = np.linalg.norm(self.ball_pos - self.ai_teammate_pos)
            if dist_ai_2 < (p_radius + b_radius):
                self._kick_ball(self.ai_teammate_pos, self.ai_teammate_vel, action_ai_2)
                reward = 1 
                
                # Pass Reward (Receive by Teammate)
                if self.last_touch_player == 1:
                    # Reward AI 1 for successful pass
                    reward += config.REWARD_PASS
                    
                self.last_touch_player = 2
        
        # Bot 1
        dist_bot = np.linalg.norm(self.ball_pos - self.bot_pos)
        if dist_bot < (p_radius + b_radius):
            self._kick_ball(self.bot_pos, self.bot_vel, action_bot)
            self.last_touch_player = 3
            
        # Bot 2
        if self.level == 4:
            dist_bot_2 = np.linalg.norm(self.ball_pos - self.bot_teammate_pos)
            if dist_bot_2 < (p_radius + b_radius):
                self._kick_ball(self.bot_teammate_pos, self.bot_teammate_vel, action_bot_2)
                self.last_touch_player = 4
                
        return reward, done, score

    def _get_action_vector(self, action):
        move_vec = np.array([0.0, 0.0])
        if action is None: return move_vec
        if np.argmax(action) == 1: move_vec[1] = -1
        elif np.argmax(action) == 2: move_vec[1] = 1
        elif np.argmax(action) == 3: move_vec[0] = -1
        elif np.argmax(action) == 4: move_vec[0] = 1
        return move_vec

    def _get_bot_heuristic_action_vector(self, bot_pos, ball_pos):
        move_vec_bot = np.array([0.0, 0.0])
        if ball_pos[0] < bot_pos[0]: move_vec_bot[0] = -1
        if ball_pos[0] > bot_pos[0]: move_vec_bot[0] = 1
        if ball_pos[1] < bot_pos[1]: move_vec_bot[1] = -1
        if ball_pos[1] > bot_pos[1]: move_vec_bot[1] = 1
        move_vec_bot *= 0.5 # Slower heuristic bot
        return move_vec_bot

    def _decode_action(self, action_idx):
        if action_idx is None: return None, 0
        # Action is 0-23
        # Physical: 0-5 (Action // 4)
        # Comm: 0-3 (Action % 4)
        
        # If action is a one-hot vector (from model prediction sometimes), convert to int
        if isinstance(action_idx, list) or isinstance(action_idx, np.ndarray):
             action_idx = np.argmax(action_idx)
             
        physical = action_idx // config.COMM_SIZE
        comm = action_idx % config.COMM_SIZE
        
        # Re-encode physical to one-hot for _move_entity helper if needed, 
        # but _move_entity helper expects the full one-hot or index?
        # _get_action_vector expects the one-hot of size 6 usually.
        # Let's make a fake one-hot for physical
        physical_one_hot = [0] * 6
        physical_one_hot[physical] = 1
        
        return physical_one_hot, comm

    def play_step(self, action_ai, action_bot=None, action_ai_2=None, action_bot_2=None):
        self.frame_iteration += 1
        
        # Decode Actions
        phys_ai, comm_ai = self._decode_action(action_ai)
        phys_ai_2, comm_ai_2 = self._decode_action(action_ai_2)
        
        # Update Comm State (Available for NEXT step)
        self.comm_ai = comm_ai
        self.comm_ai_teammate = comm_ai_2 # Teammate's message becomes my input next frame
        
        # 1. Collect User Input / Move
        # AI Team
        self.ai_pos, self.ai_vel = self._move_entity(self.ai_pos, self.ai_vel, self._get_action_vector(phys_ai))
        if self.level == 4:
             self.ai_teammate_pos, self.ai_teammate_vel = self._move_entity(self.ai_teammate_pos, self.ai_teammate_vel, self._get_action_vector(phys_ai_2))
        
        # Bot Team
        bot_move_vec = self._get_action_vector(action_bot) if action_bot is not None else self._get_bot_heuristic_action_vector(self.bot_pos, self.ball_pos)
        self.bot_pos, self.bot_vel = self._move_entity(self.bot_pos, self.bot_vel, bot_move_vec)
        if self.level == 4:
             bot_teammate_move_vec = self._get_action_vector(action_bot_2) if action_bot_2 is not None else self._get_bot_heuristic_action_vector(self.bot_teammate_pos, self.ball_pos)
             self.bot_teammate_pos, self.bot_teammate_vel = self._move_entity(self.bot_teammate_pos, self.bot_teammate_vel, bot_teammate_move_vec)
        
        # Ball Physics
        goal_status = self._move_ball()
        
        # 2. Check Collisions & Goal
        reward, done, score = self._check_collisions(phys_ai, action_bot, phys_ai_2, action_bot_2)
        
        # 5. Calculate Reward
        reward -= 0.05 # Time penalty
        game_over = False
        
        # Match Timer
        self.match_time -= 1
        if self.match_time <= 0:
            game_over = True
            
        if goal_status == 1: # AI Scored
            reward = 10
            self.score_ai += 1
            
            # Assist Reward
            # If Teammate scored (how do we know? We don't track who kicked last into goal explicitly, 
            # but last_touch_player should be 1 or 2).
            # If last_touch_player was 2 (Teammate), and previous was 1 (AI), that's an assist?
            # No, last_touch_player is updated on collision.
            # If Teammate scored, last_touch_player IS 2.
            # But we want to reward AI (Player 1) if they assisted.
            # We need to track "Assister".
            # For simplicity: If Teammate scores, and AI touched it recently? 
            # Let's just say: If Teammate scores (last_touch == 2), we give AI a small bonus?
            # Or better: If AI scores (last_touch == 1) and previous touch was 2 -> Assist for Teammate (not useful for P1 training unless shared reward).
            # If Teammate scores (last_touch == 2) and previous touch was 1 -> Assist for AI.
            # We need history of touches.
            pass # Complex to implement perfectly without history.
            # Simplified Assist: If AI scores, great. If Teammate scores, also reward AI?
            # Yes, cooperative reward.
            if self.last_touch_player == 2:
                 reward += config.REWARD_ASSIST # AI gets reward when teammate scores too!
            
            if self.score_ai >= config.GOAL_LIMIT:
                game_over = True
            else:
                self._reset_positions() # Kick-off
                # Give ball to Bot (conceded)
                self.ball_pos = np.array([self.w/2, self.h/2])
                self.ball_vel = np.array([2.0, 0.0]) # Slight push towards AI
                
            return reward, game_over, self.score_ai
            
        elif goal_status == 2: # Bot Scored
            reward = -10
            self.score_bot += 1
            if self.score_bot >= config.GOAL_LIMIT:
                game_over = True
            else:
                self._reset_positions() # Kick-off
                # Give ball to AI (conceded)
                self.ball_pos = np.array([self.w/2, self.h/2])
                self.ball_vel = np.array([-2.0, 0.0]) # Slight push towards Bot
                
            return reward, game_over, self.score_ai
            
        # Shaping: Distance to ball
        dist_to_ball = np.linalg.norm(self.ai_pos - self.ball_pos)
        if dist_to_ball < self.last_dist_to_ball:
            reward += 0.1
        else:
            reward -= 0.1
        self.last_dist_to_ball = dist_to_ball
        
        # 6. Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score_ai

    def _update_ui(self):
        if not self.render_mode:
            return
            
        # Draw Field
        if self.assets_loaded:
            self.display.blit(self.field_img, (0, 0))
        else:
            self.display.fill(BLACK)
            pygame.draw.rect(self.display, GRAY, (0, 0, self.w, self.h), 2)
            pygame.draw.line(self.display, GRAY, (self.w/2, 0), (self.w/2, self.h), 2)
            pygame.draw.circle(self.display, GRAY, (self.w/2, self.h/2), 50, 2)
        
        # Draw Goals
        goal_top = self.h/3
        goal_h = self.h/3
        pygame.draw.rect(self.display, NEON_GREEN, (0, goal_top, 5, goal_h)) # Left Goal
        pygame.draw.rect(self.display, RED, (self.w-5, goal_top, 5, goal_h)) # Right Goal
        
        # Particles
        self._update_particles()
        
        # Draw Entities
        if self.assets_loaded:
            # AI
            self._draw_sprite(self.player_img, self.ai_pos, BLUE)
            # Bot
            self._draw_sprite(self.player_img, self.bot_pos, RED)
            # Ball
            ball_rect = self.ball_img.get_rect(center=self.ball_pos.astype(int))
            # Rotate ball based on spin? Too complex for now, just blit
            self.display.blit(self.ball_img, ball_rect)
            
            if self.level == 4:
                 self._draw_sprite(self.player_img, self.ai_teammate_pos, (100, 100, 255))
                 self._draw_sprite(self.player_img, self.bot_teammate_pos, (255, 100, 100))
        else:
            pygame.draw.circle(self.display, BLUE, self.ai_pos.astype(int), 15) # AI
            pygame.draw.circle(self.display, RED, self.bot_pos.astype(int), 15) # Bot
            pygame.draw.circle(self.display, WHITE, self.ball_pos.astype(int), 10) # Ball
            
            if self.level == 4:
                 pygame.draw.circle(self.display, (100, 100, 255), self.ai_teammate_pos.astype(int), 15)
                 pygame.draw.circle(self.display, (255, 100, 100), self.bot_teammate_pos.astype(int), 15)
        
        # Draw Score & Time
        time_sec = int(self.match_time / config.FPS)
        mins = time_sec // 60
        secs = time_sec % 60
        
        text = font.render(f"AI: {self.score_ai}  Bot: {self.score_bot}   Time: {mins:02}:{secs:02}", True, WHITE)
        self.display.blit(text, [self.w/2 - text.get_width()/2, 10])
        
        pygame.display.flip()

    def _draw_sprite(self, img, pos, color):
        # Tint image
        tinted = img.copy()
        tinted.fill(color, special_flags=pygame.BLEND_MULT)
        rect = tinted.get_rect(center=pos.astype(int))
        self.display.blit(tinted, rect)

    def _update_particles(self):
        # Add particles when running (simple trail)
        if random.random() < 0.2:
             self.particles.append([self.ai_pos[0], self.ai_pos[1] + 10, 0, 0, 20, (200, 200, 200)])
             
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1 # Life
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                pygame.draw.circle(self.display, p[5], (int(p[0]), int(p[1])), 2)
        
    def get_state(self):
        # 14 Values (1v1) -> 26 Values (2v2)
        # AI Pos (2), AI Vel (2)
        # Bot Pos (2), Bot Vel (2)
        # Ball Pos (2), Ball Vel (2)
        # Extra context (2)
        # AI 2 Pos (2), AI 2 Vel (2)
        # Bot 2 Pos (2), Bot 2 Vel (2)
        
        state = [
            self.ai_pos[0]/self.w, self.ai_pos[1]/self.h,
            self.ai_vel[0]/config.MAX_SPEED, self.ai_vel[1]/config.MAX_SPEED,
            self.bot_pos[0]/self.w, self.bot_pos[1]/self.h,
            self.bot_vel[0]/config.MAX_SPEED, self.bot_vel[1]/config.MAX_SPEED,
            self.ball_pos[0]/self.w, self.ball_pos[1]/self.h,
            self.ball_vel[0]/config.BALL_MAX_SPEED, self.ball_vel[1]/config.BALL_MAX_SPEED,
            # Extra context
            (self.ball_pos[0] - self.ai_pos[0])/self.w,
            (self.ball_pos[1] - self.ai_pos[1])/self.h
        ]
        
        if self.level == 4:
            # Add 2v2 specific state
            state.extend([
                self.ai_teammate_pos[0]/self.w, self.ai_teammate_pos[1]/self.h,
                self.ai_teammate_vel[0]/config.MAX_SPEED, self.ai_teammate_vel[1]/config.MAX_SPEED,
                self.bot_teammate_pos[0]/self.w, self.bot_teammate_pos[1]/self.h,
                self.bot_teammate_vel[0]/config.MAX_SPEED, self.bot_teammate_vel[1]/config.MAX_SPEED,
                # Communication: What is my teammate saying?
                # Normalize by COMM_SIZE
                self.comm_ai_teammate / config.COMM_SIZE
            ])
        else:
            # Padding for consistency if we want fixed size, or just return smaller state?
            # Model expects fixed size. We should pad with zeros if not in 2v2, 
            # OR we use a different model for 2v2. 
            # For simplicity, let's pad with -1 (off field)
            state.extend([-1]*9) # 8 pos/vel + 1 comm
            
        return np.array(state, dtype=np.float32)
