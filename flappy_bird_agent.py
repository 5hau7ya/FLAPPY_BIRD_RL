import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import os

LR = 1e-3
GAMMA = 0.99
TAU = 1e-3
BATCH_SIZE = 64
MAX_MEMORY = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

WIDTH, HEIGHT = 288, 512
FPS = 60

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    def __len__(self):
        return len(self.memory)

class FlappyEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Flappy Bird RL Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 25, bold=True)
        self.bird_img = self.load_image('bird.png', (34, 24))
        self.pipe_img = self.load_image('pipe.png', (52, 320))
        self.bg_img = self.load_image('bg.png', (WIDTH, HEIGHT))
        self.reset()
    def load_image(self, path, size):
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, size)
        return None
    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_x = 50
        self.velocity = 0
        self.pipes = []
        self.score = 0
        self.frame_iteration = 0
        self.spawn_pipe()
        return self.get_state()
    def spawn_pipe(self):
        gap = 130
        top_height = random.randint(50, HEIGHT - 150 - gap)
        self.pipes.append({
            'x': WIDTH,
            'top': top_height,
            'bottom': top_height + gap,
            'passed': False
        })
    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        if action == 1:
            self.velocity = -8 
        self.velocity += 1
        self.velocity = min(self.velocity, 10)
        self.bird_y += self.velocity
        for pipe in self.pipes:
            pipe['x'] -= 4
        if len(self.pipes) > 0 and self.pipes[-1]['x'] < WIDTH - 150:
            self.spawn_pipe()
        if len(self.pipes) > 0 and self.pipes[0]['x'] < -60:
            self.pipes.pop(0)
        reward = 0.1
        done = False
        if self.bird_y > HEIGHT - 50 or self.bird_y < 0:
            reward = -10
            done = True
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, 34, 24)
        for pipe in self.pipes:
            top_rect = pygame.Rect(pipe['x'], 0, 52, pipe['top'])
            bottom_rect = pygame.Rect(pipe['x'], pipe['bottom'], 52, HEIGHT - pipe['bottom'])
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                reward = -10
                done = True
            if not pipe['passed'] and pipe['x'] + 52 < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                reward = 10
        self.render()
        return self.get_state(), reward, done, self.score
    def get_state(self):
        closest_pipe = next((p for p in self.pipes if p['x'] + 52 > self.bird_x), None)
        if not closest_pipe:
            return np.array([self.bird_y/HEIGHT, self.velocity/10, 1.0, 0.5, 0.5], dtype=np.float32)
        state = [
            self.bird_y / HEIGHT,
            self.velocity / 10.0,
            (closest_pipe['x'] - self.bird_x) / WIDTH,
            closest_pipe['top'] / HEIGHT,
            closest_pipe['bottom'] / HEIGHT
        ]
        return np.array(state, dtype=np.float32)
    def render(self):
        if self.bg_img: self.screen.blit(self.bg_img, (0, 0))
        else: self.screen.fill((112, 197, 206)) # Original Sky Blue
        for pipe in self.pipes:
            if self.pipe_img:
                top_img = pygame.transform.flip(self.pipe_img, False, True)
                self.screen.blit(top_img, (pipe['x'], pipe['top'] - 320))
                self.screen.blit(self.pipe_img, (pipe['x'], pipe['bottom']))
            else:
                pygame.draw.rect(self.screen, (116, 192, 42), (pipe['x'], 0, 52, pipe['top']))
                pygame.draw.rect(self.screen, (116, 192, 42), (pipe['x'], pipe['bottom'], 52, HEIGHT - pipe['bottom']))
        if not self.bg_img:
            pygame.draw.rect(self.screen, (222, 216, 149), (0, HEIGHT - 50, WIDTH, 50))
        if self.bird_img:
            self.screen.blit(self.bird_img, (self.bird_x, self.bird_y))
        else:
            pygame.draw.ellipse(self.screen, (234, 223, 56), (self.bird_x, self.bird_y, 34, 24))
        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)

plt.ion()
def plot_performance(scores, mean_scores):
    plt.figure(1)
    plt.clf()
    plt.title('Agent Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score per Game', color='lightblue')
    plt.plot(mean_scores, label='10-Game Moving Avg', color='orange', linewidth=2)
    plt.legend()
    plt.ylim(ymin=0)
    plt.pause(0.001)

def train():
    device = torch.device("cpu")
    print(f"Training on: {device}")
    env = FlappyEnv()
    state_dim = 5
    action_dim = 2
    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(MAX_MEMORY)
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    epsilon = EPSILON_START
    n_games = 0
    while True:
        state = env.reset()
        done = False
        game_score = 0
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = policy_net(state_tensor).argmax().item()
            next_state, reward, done, score = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE) 
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                    next_targets = target_net(next_states).gather(1, next_actions)
                    target_q = rewards + (GAMMA * next_targets * (1 - dones))
                current_q = policy_net(states).gather(1, actions)
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
        n_games += 1
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        scores.append(score)
        total_score += score
        if len(scores) >= 10:
            mean_score = sum(scores[-10:]) / 10
        else:
            mean_score = total_score / n_games
        mean_scores.append(mean_score)
        if score > record:
            record = score
            torch.save(policy_net.state_dict(), 'best_model.pth')
        print(f"Game: {n_games} | Score: {score} | Record: {record} | Epsilon: {epsilon:.2f}")
        plot_performance(scores, mean_scores)
if __name__ == '__main__':
    train()