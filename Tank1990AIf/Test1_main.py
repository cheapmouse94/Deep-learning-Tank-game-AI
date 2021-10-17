import threading
import pygame
import time
import sys
import os
from pygame.locals import *
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
from Tank_AI import Linear_QNet, QTrainer
import random

FPS = 1000
SQM = 64
EAGLE_Y = []
EAGLE_G = []
BULLETS_Y_objects = []
BULLETS_Y_RECT = []
BULLETS_G_objects = []
BULLETS_G_RECT = []
BACKGROUND_RECT = []
GRASS_RECT = []
WATER_RECT = []
BRICK_RECT = []
BRICK_RECT_MANY = []
BRICK_RECT_MINI = []
SOLID_RECT = []
MAPPING = [
'HHHHHHHHHHHHHHHHH',
'HHHHHHHHHHHHHHHHH',
'HHHHSGOOOBOOSGOHH',
'HHHHGBOWBGBOOBGHH',
'HHHHOG1BGSGB2GOHH',
'HHHHGBOOBGBWOBGHH',
'HHHHOGSOOBOOOGSHH',
'HHHHHHHHHHHHHHHHH',
'HHHHHHHHHHHHHHHHH'
]

TANK_YELLOW_IMG = [pygame.transform.scale((pygame.image.load(os.path.join('textures', 'yellow_tank_up.png'))), (52,52)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'yellow_tank_down.png'))), (52,52)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'yellow_tank_left.png'))), (52,52)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'yellow_tank_right.png'))), (52,52))]

TANK_GREEN_IMG = [pygame.transform.scale((pygame.image.load(os.path.join('textures', 'green_tank_up.png'))), (52,52)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'green_tank_down.png'))), (52,52)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'green_tank_left.png'))), (52,52)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'green_tank_right.png'))), (52,52))]

BULLET_IMG = [pygame.transform.scale((pygame.image.load(os.path.join('textures', 'bullet_u.png'))), (16,22)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'bullet_d.png'))), (16,22)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'bullet_l.png'))), (22,16)),
pygame.transform.scale((pygame.image.load(os.path.join('textures', 'bullet_r.png'))), (22,16))]

WATER_1_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'prop_water_1.png'))), (64,64))
WATER_2_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'prop_water_2.png'))), (64,64))
BRICK_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'prop_brick.png'))), (64,64))
BRICK_IMG_MINI = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'prop_brick_mini.png'))), (32,32))
GRASS_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'prop_grass.png'))), (64,64))
SOLIDWALL_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'prop_solid_wall.png'))), (64,64))
EAGLE_1_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'entity_eagle_1.png'))), (64,64))
EAGLE_2_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'entity_eagle_2.png'))), (64,64))
EXPLOSION_1_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'entity_explosion_1.png'))), (64,64))
EXPLOSION_2_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'entity_explosion_2.png'))), (64,64))
EXPLOSION_3_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'entity_explosion_3.png'))), (64,64))
EXPLOSION_GREAT_1_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'entity_explosion_great_1.png'))), (128,128))
EXPLOSION_GREAT_2_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'entity_explosion_great_2.png'))), (128,128))
INVICIBLE_1_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'invicible_1.png'))), (52,52))
INVICIBLE_2_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'invicible_2.png'))), (52,52))
BACKGROUND_IMG = pygame.transform.scale((pygame.image.load(os.path.join('textures', 'background.png'))), (64,64))

MAX_MEMORY = 100_000_000
BATCH_SIZE = 1000
LR = 0.0001

class AI_YELLOW:
    def __init__(self):
        self.state = []
        self.gamma = 0.5
        self.score = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(24, 256, 64, 5)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, a, b, c, d, e, f, g, h, i, j):
        self.state = []
        self.state_n = [a, b, c, d, e, f, g, h, i, j]
        for n in self.state_n:
            for mn in n:
                self.get_state_loop(mn)      
        return self.state 
        
    def get_state_loop(self, m):
        self.state.append(m)

    def get_action(self, state, frame):
        final_move = [0,0,0,0,0]
        if frame > 500:
            state0 = torch.tensor(state, dtype=float)
            state0 = state0.double()
            prediction = self.model(state0.float())
            move = torch.argmax(prediction).item()
            move_0 = torch.softmax(prediction, dim=-1).detach().numpy()
            x = random.choices([0,1,2,3,4],move_0)
            final_move[move] = 1
        else:
            rand = random.randint(0,4)
            final_move[rand] = 1
        return final_move
    
    def print_state(self, state, frame, score):
        if frame % 100 == 0:
            print(f'---ŻÓŁTY------klata nr. {frame}--------wynik sumaryczny {score}---------')
            print(len(state))
            print(f'Pozycja Zółtego czołgu względem Zielonego czołgu {state[0:4]}')
            #print(f'Pozycja Zółtego czołgu względem własnego orła {state[4:8]}')
            #print(f'Pozycja Zółtego czołgu względem obcego orła {state[8:12]}')
            print(f'Zwrot swojego czołgu {state[4:8]}')
            print(f'Obecność swojego pocisku {state[8]}')
            print(f'Obecność przeciwnika pocisku {state[9]}')
            print(f'Kierunek swojego pocisku {state[10:14]}')
            print(f'Kierunek przeciwnika pocisku {state[14:18]}')
            print(f'Zwrot czołgu do obiektów 1.Tło - {state[18]} 2.Ściana - {state[19]} 3.Orzeł własny - ??? 4.Orzeł przeciwnika - ??? 5.Przeciwnik - {state[20]}')
            print(f'Czy Żółty czołg utkną? {state[21]}')
            print(f'Czy zielony czołg otrzymał obrażenia? {state[22]}')
            print(f'Czy żółty czołg otrzymał obrażenia? {state[23]}')
            #print(f'Czy orzeł zółtego otrzymał obrażenia przez żółtego? {state[23]}')
            #print(f'Czy orzeł zielonego otrzymał obrażenia przez żółtego? {state[24]}')
            print('------------------------------------------------------------')

    def train_short_memory(self, satte_old, action, reward, nest_state, done):
        self.trainer.train_step(satte_old, action, reward, nest_state, done)

    def remember(self, satte_old, action, reward, nest_state, done):
        self.memory.append((satte_old, action, reward, nest_state, done))
    
    def final_score(self, reward):
        self.score += reward
        return "{0:0.2f}".format(self.score)

class AI_GREEN:
    def __init__(self):
        self.state = []
        self.gamma = 0.5
        self.score = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(24, 256, 64, 5)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, a, b, c, d, e, f, g, h, i, j):
        self.state = []
        self.state_n = [a, b, c, d, e, f, g, h, i, j]
        for n in self.state_n:
            for mn in n:
                self.get_state_loop(mn)      
        return self.state 
        
    def get_state_loop(self, m):
        self.state.append(m)

    def get_action(self, state, frame):
        final_move = [0,0,0,0,0]
        if frame > 500:
            state0 = torch.tensor(state, dtype=float)
            state0 = state0.double()
            prediction = self.model(state0.float())
            move = torch.argmax(prediction).item()
            move_0 = torch.softmax(prediction, dim=-1).detach().numpy()
            x = random.choices([0,1,2,3,4],move_0)
            final_move[move] = 1
        else:
            rand = random.randint(0,4)
            final_move[rand] = 1
        return final_move
    
    def print_state(self, state, frame, score):
        if frame % 100 == 0:
            print(f'---ZIELONY------klata nr. {frame}--------wynik sumaryczny {score}---------')
            print(len(state))
            print(f'Pozycja Zielonego czołgu względem Zółtego czołgu {state[0:4]}')
            #print(f'Pozycja Zielonego czołgu względem własnego orła {state[4:8]}')
            #print(f'Pozycja Zielonego czołgu względem obcego orła {state[8:12]}')
            print(f'Zwrot swojego czołgu {state[4:8]}')
            print(f'Obecność swojego pocisku {state[8]}')
            print(f'Obecność przeciwnika pocisku {state[9]}')
            print(f'Kierunek swojego pocisku {state[10:14]}')
            print(f'Kierunek przeciwnika pocisku {state[14:18]}')
            print(f'Zwrot czołgu do obiektów 1.Tło - {state[18]} 2.Ściana - {state[19]} 3.Orzeł własny - ??? 4.Orzeł przeciwnika - ??? 5.Przeciwnik - {state[20]}')
            print(f'Czy Zielony czołg utkną? {state[21]}')
            print(f'Czy Zółty czołg otrzymał obrażenia? {state[22]}')
            print(f'Czy Zielony czołg otrzymał obrażenia? {state[23]}')
            #print(f'Czy orzeł zielonego otrzymał obrażenia przez zielonego? {state[32]}')
            #print(f'Czy orzeł żółtego otrzymał obrażenia przez zielonego? {state[33]}')
            print('------------------------------------------------------------')

    def train_short_memory(self, satte_old, action, reward, nest_state, done):
        self.trainer.train_step(satte_old, action, reward, nest_state, done)

    def remember(self, satte_old, action, reward, nest_state, done):
        self.memory.append((satte_old, action, reward, nest_state, done))

    def final_score(self, reward):
        self.score += reward
        return "{0:0.2f}".format(self.score)

class On_Hit_By_Yellow:
    def __init__(self, dir):
        self.dir = dir
        self.x_exp = 0
        self.y_exp = 0
        self.frame_l = 0
        self.frame_h = 0
        self.break_bullet_one_time_flag = True
        self.allow_explosion_little = False
        self.allow_explosion_hard = False


    def brick_on_hit(self, i, e):
        BRICK_RECT_TEMP = []
        for b in BRICK_RECT_MINI:
            if e.colliderect(b):
                BRICK_RECT_TEMP.append(b)
        if len(BRICK_RECT_TEMP) >= 1:
            for x in BRICK_RECT_TEMP:
                BRICK_RECT_MINI.remove(x)
            self.explosion_find_location()
            self.allow_explosion_hard = True
            return True
        return False

    def solid_on_hit(self, i, e):
        for b in SOLID_RECT:
            if e.colliderect(b):
                self.explosion_find_location()
                self.allow_explosion_little = True
                return True
        return False

    def background_on_hit(self, i, e):
        for b in BACKGROUND_RECT:
            if e.colliderect(b):
                self.explosion_find_location()
                self.allow_explosion_little = True
                return True
        return False
    def green_tank_on_hit(self, i, e, TG_MASK, TG_CLASS, TG_DEST, TG_INVI):
        if e.colliderect(TG_MASK) and TG_INVI is False:
            print('Green Tank took damage')
            self.does_enemy_tank_got_hit = True
            TG_CLASS.__init__()
            return True
        return False

    def eagle_greens_tank_on_hit(self, i, e, TG_CLASS, TY_CLASS, MAPPING):
        for b in EAGLE_G:
            if e.colliderect(b):
                TG_CLASS.__init__()
                TY_CLASS.__init__()
                print('Green\'s eagle gas been destroyed')
                self.does_enemy_eagle_got_hit = True
                return True
        return False

    def eagle_yellows_tank_on_hit(self, i, e, TG_CLASS, TY_CLASS, MAPPING):
        for b in EAGLE_Y:
            if e.colliderect(b):
                TG_CLASS.__init__()
                TY_CLASS.__init__()
                print('Yellow\'s eagle gas been destroyed')
                self.does_ally_eagle_fot_hit = True
                return True
        return False

    def enemys_bullet_on_hit(self, i, e):
        for b in BULLETS_G_RECT:
            if e.colliderect(b):
                if len(BULLETS_G_RECT) >= 1:
                    BULLETS_G_objects.pop(i)
                    BULLETS_G_RECT.pop(i)
                    return True
        return False                

    def break_bullet(self, i):
        if self.break_bullet_one_time_flag:
            BULLETS_Y_objects.pop(i)
            BULLETS_Y_RECT.pop(i)
            self.break_bullet_one_time_flag = False

    def explosion_find_location(self):
        for k in BULLETS_Y_RECT:
            if self.dir == 'right':
                self.x_exp = k.x 
                self.y_exp = k.y - 26
            if self.dir == 'left':
                self.x_exp = k.x
                self.y_exp = k.y - 26
            if self.dir == 'up':
                self.x_exp = k.x - 26
                self.y_exp = k.y
            if self.dir == 'down':
                self.x_exp = k.x - 26
                self.y_exp = k.y 
        

    def draw_explosion_little(self, screen, elf):
        if self.allow_explosion_little and elf:
            if self.frame_l == 0:
                screen.blit(EXPLOSION_1_IMG,(self.x_exp, self.y_exp))
            if self.frame_l == 1:
                screen.blit(EXPLOSION_2_IMG,(self.x_exp, self.y_exp))
            if self.frame_l == 2:
                screen.blit(EXPLOSION_1_IMG,(self.x_exp, self.y_exp))
            if self.frame_l >= 2:
                self.allow_explosion_little = False
                elf = False
                self.frame_l += 0
            else:
                self.frame_l += 1

    def draw_explosion_hard(self, screen, ehf):
        if self.allow_explosion_hard and ehf:
            if self.frame_h <= 1:
                screen.blit(EXPLOSION_2_IMG,(self.x_exp, self.y_exp))
            if self.frame_h >= 2 and self.frame_h < 4:
                screen.blit(EXPLOSION_3_IMG,(self.x_exp, self.y_exp))
            if self.frame_h >= 4:
                ehf = False
                self.allow_explosion_hard = False
                self.frame_h = 0
            else:
                self.frame_h += 1

class On_Hit_By_Green:
    def __init__(self, dir):
        self.dir = dir
        self.x_exp = 0
        self.y_exp = 0
        self.frame_l = 0
        self.frame_h = 0
        self.break_bullet_one_time_flag = True
        self.allow_explosion_little = False
        self.allow_explosion_hard = False

    def brick_on_hit(self, i, e):
        BRICK_RECT_TEMP = []
        for b in BRICK_RECT_MINI:
            if e.colliderect(b):
                BRICK_RECT_TEMP.append(b)
        if len(BRICK_RECT_TEMP) >= 1:
            for x in BRICK_RECT_TEMP:
                BRICK_RECT_MINI.remove(x)
            self.explosion_find_location()
            self.allow_explosion_hard = True
            return True
        return False

    def solid_on_hit(self, i, e):
        for b in SOLID_RECT:
            if e.colliderect(b):
                self.explosion_find_location()
                self.allow_explosion_little = True
                return True
        return False

    def background_on_hit(self, i, e):
        for b in BACKGROUND_RECT:
            if e.colliderect(b):
                self.explosion_find_location()
                self.allow_explosion_little = True
                return True
        return False

    def yellow_tank_on_hit(self, i, e, TY_MASK, TG_CLASS, TY_DEST, TY_INVI):
        if e.colliderect(TY_MASK) and TY_INVI is False:
            TY_DEST = True
            TG_CLASS.__init__()
            print('Yellow Tank took damage')
            self.does_enemy_tank_got_hit = True
            return True
        return False

    def eagle_greens_tank_on_hit(self, i, e, TG_CLASS, TY_CLASS, MAPPING):
        for b in EAGLE_G:
            if e.colliderect(b):
                TG_CLASS.__init__()
                TY_CLASS.__init__()
                print('Green\'s eagle has been destroyed')
                self.does_ally_eagle_got_hit = True
                return True
        return False

    def eagle_yellows_tank_on_hit(self, i, e, TG_CLASS, TY_CLASS, MAPPING):
        for b in EAGLE_Y:
            if e.colliderect(b): 
                TG_CLASS.__init__()
                TY_CLASS.__init__()
                print('Yellow\'s eagle has been destroyed')
                self.does_enemy_eagle_got_hit = True
                return True
        return False

    def enemys_bullet_on_hit(self, i, e):
        for b in BULLETS_Y_RECT:
            if e.colliderect(b):
                if len(BULLETS_Y_RECT) >= 1:
                    BULLETS_Y_objects.pop(i)
                    BULLETS_Y_RECT.pop(i) 
                    return True
        return False     

    def break_bullet(self, i):
        if self.break_bullet_one_time_flag:
            BULLETS_G_objects.pop(i)
            BULLETS_G_RECT.pop(i)
            self.break_bullet_one_time_flag = False

    def explosion_find_location(self):
        for k in BULLETS_G_RECT:
            if self.dir == 'right':
                self.x_exp = k.x 
                self.y_exp = k.y - 26
            if self.dir == 'left':
                self.x_exp = k.x
                self.y_exp = k.y - 26
            if self.dir == 'up':
                self.x_exp = k.x - 26
                self.y_exp = k.y 
            if self.dir == 'down':
                self.x_exp = k.x - 26
                self.y_exp = k.y 
        
    def draw_explosion_little(self, screen, elf):
        if self.allow_explosion_little and elf:
            if self.frame_l == 0:
                screen.blit(EXPLOSION_1_IMG,(self.x_exp, self.y_exp))
            if self.frame_l == 1:
                screen.blit(EXPLOSION_2_IMG,(self.x_exp, self.y_exp))
            if self.frame_l == 2:
                screen.blit(EXPLOSION_1_IMG,(self.x_exp, self.y_exp))
            if self.frame_l >= 2:
                self.allow_explosion_little = False
                elf = False
                self.frame_l += 0
            else:
                self.frame_l += 1
    def draw_explosion_hard(self, screen, ehf):
        if self.allow_explosion_hard and ehf:
            if self.frame_h == 0:
                screen.blit(EXPLOSION_2_IMG,(self.x_exp, self.y_exp))
            if self.frame_h == 1:
                screen.blit(EXPLOSION_3_IMG,(self.x_exp, self.y_exp))
            if self.frame_h == 2:
                screen.blit(EXPLOSION_2_IMG,(self.x_exp, self.y_exp))
            if self.frame_h >= 2:
                ehf = False
                self.allow_explosion_hard = False
                self.frame_h = 0
            else:
                self.frame_h += 1

class Mapping:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.frames = 0
        self.convert_entities()

    def convert_entities(self):
        for row in MAPPING:
            for col in row:
                if col == 'H':
                    BACKGROUND_RECT.append(pygame.Rect((self.x,self.y,SQM,SQM)))
                elif col == 'G':
                    GRASS_RECT.append(pygame.Rect((self.x,self.y,SQM,SQM)))
                elif col == 'W':
                    WATER_RECT.append(pygame.Rect((self.x,self.y,SQM,SQM)))
                elif col == 'B':

                    #BRICK_RECT.append(pygame.Rect((self.x,self.y,SQM,SQM)))
                    #BRICK_RECT_MANY.append(BRICK_IMG)

                    #self.convert_entities_mini()
                    pass
                elif col == 'S':
                    SOLID_RECT.append(pygame.Rect((self.x,self.y,SQM,SQM)))
                elif col == '3':
                    EAGLE_Y.append(pygame.Rect((self.x,self.y,SQM,SQM)))
                elif col == '4':
                    EAGLE_G.append(pygame.Rect((self.x,self.y,SQM,SQM)))
                self.x+=SQM
            self.y+=SQM
            self.x=0
     
    def convert_entities_mini(self):
        self.x_mini = self.x
        self.y_mini = self.y
        for i in range(2):
            for j in range(2):
                BRICK_RECT_MINI.append(pygame.Rect((self.x_mini,self.y_mini,SQM/2,SQM/2)))
                self.x_mini += SQM/2
            self.y_mini += SQM/2
            self.x_mini = self.x



    def draw_props(self, screen):
        for x in BACKGROUND_RECT:
            #pygame.draw.rect(screen,(89, 89, 89),x)
            screen.blit(BACKGROUND_IMG, (x.x,x.y))
        for x in GRASS_RECT:
            #pygame.draw.rect(screen,(51, 204, 51),x)
            screen.blit(GRASS_IMG, (x.x,x.y))
        for x in WATER_RECT:
            #pygame.draw.rect(screen,(0, 153, 255),x)
            if self.frames <= 30:
                screen.blit(WATER_1_IMG, (x.x,x.y))
            else:
                screen.blit(WATER_2_IMG, (x.x,x.y))
        '''
        for x in BRICK_RECT:
            screen.blit(BRICK_IMG, (x.x,x.y))
        
        for x in BRICK_RECT_MINI:
            screen.blit(BRICK_IMG_MINI, (x.x,x.y))
            '''
        for x in SOLID_RECT:
            screen.blit(SOLIDWALL_IMG, (x.x,x.y))
        for x in EAGLE_Y:
            screen.blit(EAGLE_1_IMG, (x.x,x.y))
        for x in EAGLE_G:
            screen.blit(EAGLE_1_IMG, (x.x,x.y))
        self.frames += 1
        if self.frames == 60:
            self.frames = 0
        
class Bullet_TY(object):
    def __init__(self,x,y,dir):
        self.dir = dir
        self.x = x
        self.y = y
        self.vel = 22

        if self.dir == 'right':
            self.x = x+15
            self.y = y+18
            self.width = 22
            self.height = 16
        elif self.dir == 'left':
            self.x = x+15
            self.y = y+18
            self.width = 22
            self.height = 16
        elif self.dir == 'down':
            self.x = x+18
            self.y = y+15
            self.width = 16
            self.height = 22
        elif self.dir == 'up':
            self.x = x+18
            self.y = y+7
            self.width = 16
            self.height = 22

    def move(self):
        if self.dir == 'right':
            self.x += self.vel
        elif self.dir == 'left':
            self.x -= self.vel
        elif self.dir == 'down':
            self.y += self.vel
        elif self.dir == 'up':
            self.y -= self.vel
            
    def movehitbox(self, rect):
        if self.dir == 'right':
            rect.x += self.vel
        elif self.dir == 'left':
            rect.x -= self.vel
        elif self.dir == 'down':
            rect.y += self.vel
        elif self.dir == 'up':
            rect.y -= self.vel


    def draw(self, screen):
        if self.dir == 'right':
            self.BULLET_DRAW = BULLET_IMG[3]
        elif self.dir == 'left':
            self.BULLET_DRAW = BULLET_IMG[2]
        elif self.dir == 'down':
            self.BULLET_DRAW = BULLET_IMG[1]
        elif self.dir == 'up':
            self.BULLET_DRAW = BULLET_IMG[0]
        
        screen.blit(self.BULLET_DRAW, (self.x, self.y))
        
class Tank_Yellow:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = [False, False, False, False]
        self.TY_face = TANK_YELLOW_IMG[3]
        self.TY_face_txt = 'right'
        self.tank_yellow_shoot_allow = True
        self.tank_yellow_shoot_cooldown = False
        self.explosion_l_flag = False
        self.explosion_h_flag = False
        self.yellow_tank_destroyed = False
        self.yellow_tank_invicible = True
        self.frames_inv = 0
        self.bullet_dir = None

        self.eagle_yellows_tank_on_hit_state = False
        self.green_tank_on_hit_state = False
        self.eagle_greens_tank_on_hit_state = False

        self.AI_player = True
        self.Human_player = True

        for row in MAPPING:
            for col in row:
                if col == '1':
                    self.ty_pos_x = self.x
                    self.ty_pos_y = self.y
                self.x+=SQM
            self.y+=SQM
            self.x=0

        self.TY_mask = pygame.Rect(self.ty_pos_x, self.ty_pos_y, 52, 52)

    def bind(self, event):
        if event.type == KEYDOWN:
            if event.key == K_d:
                self.actions[0] = True
            elif event.key == K_a:
                self.actions[1] = True
            elif event.key == K_s:
                self.actions[2] = True
            elif event.key == K_w:
                self.actions[3] = True
        if event.type == KEYUP:
            if event.key == K_d:
                self.actions[0] = False
            elif event.key == K_a:
                self.actions[1] = False
            elif event.key == K_s:
                self.actions[2] = False
            elif event.key == K_w:
                self.actions[3] = False

    def move_tank(self, action):

        self.movement = [0,0]
        if action[0]:
            self.movement[0] += 8
            self.TY_face = TANK_YELLOW_IMG[3]
            self.TY_face_txt = 'right'
        elif action[1]:
            self.movement[0] -= 8
            self.TY_face = TANK_YELLOW_IMG[2]
            self.TY_face_txt = 'left'
        elif action[3]:
            self.movement[1] -= 8
            self.TY_face = TANK_YELLOW_IMG[0]
            self.TY_face_txt = 'up'
        elif action[2]:
            self.movement[1] += 8
            self.TY_face = TANK_YELLOW_IMG[1]
            self.TY_face_txt = 'down'
        self.TY_mask.x += self.movement[0]
        self.collisions_h = self.collision_test()
    
        for tile in self.collisions_h:
            if self.movement[0] > 0:
                self.TY_mask.right = tile.left
            if self.movement[0] < 0:
                self.TY_mask.left = tile.right
        
        self.TY_mask.y += self.movement[1]
        self.collisions_v = self.collision_test()
        
        for tile in self.collisions_v:
            if self.movement[1] > 0:
                self.TY_mask.bottom = tile.top
            if self.movement[1] < 0:
                self.TY_mask.top = tile.bottom

        self.collisions_sum = [self.collisions_h, self.collisions_v]

        
    def collision_test(self):
        colli = []
        for back in BACKGROUND_RECT:
            if self.TY_mask.colliderect(back):
                colli.append(back)
        for back in SOLID_RECT:
            if self.TY_mask.colliderect(back):
                colli.append(back)
        for back in BRICK_RECT:
            if self.TY_mask.colliderect(back):
                colli.append(back)
        for back in WATER_RECT:
            if self.TY_mask.colliderect(back):
                colli.append(back)
        for back in EAGLE_Y:
            if self.TY_mask.colliderect(back):
                colli.append(back)
        for back in EAGLE_G:
            if self.TY_mask.colliderect(back):
                colli.append(back)
        for back in BRICK_RECT_MINI:
            if self.TY_mask.colliderect(back):
                colli.append(back)
        return colli

    def draw(self, screen, flag_1, flag_2):
        if flag_1 is False:
            screen.blit(self.TY_face,(self.TY_mask.x,self.TY_mask.y))
        if flag_2:
            if (self.frames_inv % 4) == 0 or (self.frames_inv % 4) == 1:
                screen.blit(INVICIBLE_1_IMG,(self.TY_mask.x,self.TY_mask.y))
            elif (self.frames_inv % 4) == 2 or (self.frames_inv % 4) == 3:
                screen.blit(INVICIBLE_2_IMG,(self.TY_mask.x,self.TY_mask.y))
            if self.frames_inv >= 45:
                self.yellow_tank_invicible = False
            self.frames_inv += 1
        
    def bind_shoot(self, Flag):
        if Flag:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                flag_temp = True
                self.execute_shoot(flag_temp)

    def execute_shoot(self, Flag):
        if Flag:
            self.frames = 0
            self.tank_yellow_shoot_cooldown = True
            self.tank_yellow_shoot_allow = False
            self.b_ty = Bullet_TY(self.TY_mask.x, self.TY_mask.y, self.TY_face_txt)
            BULLETS_Y_objects.append(self.b_ty)
            BULLETS_Y_RECT.append(pygame.Rect(self.b_ty.x,self.b_ty.y,self.b_ty.width,self.b_ty.height))
            self.OHBY = On_Hit_By_Yellow(self.b_ty.dir)
            self.bullet_dir = self.b_ty.dir
              
    def shoot_delay(self, flag):
        if flag:
            if len(BULLETS_Y_RECT) == 0 and self.frames > 20:
                self.tank_yellow_shoot_allow = True
                self.tank_yellow_shoot_cooldown = False
                self.bullet_dir = None
            self.frames += 1

    def bullets_onhit(self, TG_MASK, TG_CLASS, TY_CLASS, TG_DEST, TG_INVI, MAPPING, screen):
        if len(BULLETS_Y_RECT) >= 1:
            for i, e in enumerate(BULLETS_Y_RECT):
                self.explosion_h_flag = True
                self.explosion_l_flag = True
                self.brick_on_hit_state = self.OHBY.brick_on_hit(i, e)
                self.background_on_hit_state = self.OHBY.background_on_hit(i, e)
                self.green_tank_on_hit_state = self.OHBY.green_tank_on_hit(i, e, TG_MASK, TG_CLASS, TG_DEST, TG_INVI)
                self.solid_on_hit_state = self.OHBY.solid_on_hit(i, e)
                self.eagle_greens_tank_on_hit_state = self.OHBY.eagle_greens_tank_on_hit(i, e, TG_CLASS, TY_CLASS, MAPPING)
                self.eagle_yellows_tank_on_hit_state = self.OHBY.eagle_yellows_tank_on_hit(i, e, TG_CLASS, TY_CLASS, MAPPING)
                self.enemys_bullet_on_hit_state = self.OHBY.enemys_bullet_on_hit(i, e)
                self.states = [self.brick_on_hit_state, 
                self.background_on_hit_state,
                self.green_tank_on_hit_state,
                self.solid_on_hit_state,
                self.eagle_greens_tank_on_hit_state,
                self.eagle_yellows_tank_on_hit_state,
                self.enemys_bullet_on_hit_state]
                for xi in self.states:
                    if xi:
                        self.OHBY.break_bullet(i)
        if self.explosion_l_flag or self.explosion_h_flag:
            self.OHBY.draw_explosion_little(screen, self.explosion_l_flag)
            self.OHBY.draw_explosion_hard(screen, self.explosion_h_flag)

    def yellow_tank_position_relative_with_green_tank(self, TY_mask, TG_mask):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        if TY_mask.x <= TG_mask.x:
            flags[0] = True
        if TY_mask.x >= TG_mask.x:
            flags[1] = True
        if TY_mask.y >= TG_mask.y:
            flags[2] = True
        if TY_mask.y <= TG_mask.y:
            flags[3] = True
        return flags
        
    def yellow_eagle_position_relative_with_yellow_tank(self, TY_mask):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        for i in EAGLE_Y:
            if TY_mask.x <= i.x:
                flags[0] = True
            if TY_mask.x >= i.x:
                flags[1] = True
            if TY_mask.y >= i.y:
                flags[2] = True
            if TY_mask.y <= i.y:
                flags[3] = True 
        return flags

    def green_eagle_position_relative_with_yellow_tank(self, TY_mask):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        for i in EAGLE_G:
            if TY_mask.x <= i.x:
                flags[0] = True
            if TY_mask.x >= i.x:
                flags[1] = True
            if TY_mask.y >= i.y:
                flags[2] = True
            if TY_mask.y <= i.y:
                flags[3] = True 
        return flags

    def yellow_tank_direction(self):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        if self.TY_face_txt == 'right':
            flags[0] = True
        elif self.TY_face_txt == 'left':
            flags[1] = True
        elif self.TY_face_txt == 'up':
            flags[2] = True
        elif self.TY_face_txt == 'down':
            flags[3] = True
        return flags

    def yellow_tank_bullet_presence(self):
        flag = False
        if self.tank_yellow_shoot_allow is True:
            flag = False
        elif self.tank_yellow_shoot_allow is False:
            flag = True
        return [flag]

    def yellow_tank_own_bullet_direction(self, dir, pres):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        if pres:
            if dir == 'right':
                flags[0] = True
            elif dir == 'left':
                flags[1] = True
            elif dir == 'up':
                flags[2] = True
            elif dir == 'down':
                flags[3] = True
        return flags

    def yellow_tank_faced_to_entity_solid(self, dir, TY_MASK, TG_MASK, win):
        self.xn = TY_MASK.x + 26
        self.yn = TY_MASK.y + 26
        if dir[0] is True:
            for i in range(44):
                self.xn += 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                pygame.draw.rect(win, (255, 0, 0), self.sample)
                self.loop_logic_background = self.yellow_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.yellow_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                #self.loop_logic_enemys_eagle = self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                self.loop_logic_enemy = self.yellow_tank_faced_to_enemy_loop(self.sample, TG_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]
        if dir[1] is True:
            for i in range(44):
                self.xn -= 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                pygame.draw.rect(win, (255, 0, 0), self.sample)
                self.loop_logic_background = self.yellow_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.yellow_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                #self.loop_logic_enemys_eagle = self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                self.loop_logic_enemy = self.yellow_tank_faced_to_enemy_loop(self.sample, TG_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]

        if dir[2] is True:
            for i in range(44):
                self.yn -= 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                pygame.draw.rect(win, (255, 0, 0), self.sample)
                self.loop_logic_background = self.yellow_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.yellow_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                #self.loop_logic_enemys_eagle = self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                self.loop_logic_enemy = self.yellow_tank_faced_to_enemy_loop(self.sample, TG_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]


        if dir[3] is True:
            for i in range(44):
                self.yn += 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                pygame.draw.rect(win, (255, 0, 0), self.sample)
                self.loop_logic_background = self.yellow_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.yellow_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                #self.loop_logic_enemys_eagle = self.yellow_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                self.loop_logic_enemy = self.yellow_tank_faced_to_enemy_loop(self.sample, TG_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]
        return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]
    def yellow_tank_faced_to_entity_loop(self, sample, entity):
        self.sample = sample
        for ni in entity:
            if self.sample.colliderect(ni):
                return True
        return False

    def yellow_tank_faced_to_enemy_loop(self, sample, TG_MASK):
        self.sample = sample
        if self.sample.colliderect(TG_MASK):
            return True
        return False

    def yellow_tank_stuck(self, colli):
        if len(colli[0]) >= 1 or len(colli[1]) >= 1:
            return [True]
        return [False]

    def green_tank_got_hit(self, flag):
        if self.green_tank_on_hit_state:
            self.green_tank_on_hit_state = False
            print('Żółty czołg zniszczył zielony czołg')
            return [True]
        else:
            return [False]
        
    def yellow_eagle_got_hit_by_yellow(self, flag):
        if self.eagle_yellows_tank_on_hit_state:
            self.eagle_yellows_tank_on_hit_state = False
            print('Żółty czołg zniszczył swojego orła')
            return [True]
        else:
            return [False]

    def green_eagle_got_hit_by_yellow(self, flag):
        if self.eagle_greens_tank_on_hit_state:
            self.eagle_greens_tank_on_hit_state = False
            print('Żółty czołg zniszczył orła przeciwnika')
            return [True]
        else:
            return [False]

    def yellow_tank_collision_sensor(self, TY_MASK):
        self.xs = TY_MASK.x - 2
        self.ys = TY_MASK.y - 2
        self.coli_sensor = pygame.Rect(self.xs,self.ys,56,56)
        for n in SOLID_RECT:
            if self.coli_sensor.colliderect(n):
                return [True]
        for n in WATER_RECT:
            if self.coli_sensor.colliderect(n):
                return [True]
        for n in BACKGROUND_RECT:
            if self.coli_sensor.colliderect(n):
                return [True]
        return [False]

    def play_step(self, action, green_tank_got_hit_by_yellow, yellow_tank_got_hit_by_green, yellow_eagle_got_hit_by_yellow, green_eagle_got_hit_by_yellow, yellow_tank_collision_sensor_state, frame_counter_idle):
        self.move_it(action)
        REWARD = 0
        GAME_OVER = False
        if yellow_tank_collision_sensor_state[0]:
            REWARD = - 0.1
        elif green_tank_got_hit_by_yellow[0]:
            GAME_OVER = True
            REWARD = 50
        elif yellow_tank_got_hit_by_green[0]:
            GAME_OVER = True
            REWARD = -50
        elif yellow_eagle_got_hit_by_yellow[0]:
            GAME_OVER = True
            REWARD = -150
        elif green_eagle_got_hit_by_yellow[0]:
            GAME_OVER = True
            REWARD = 150
        elif frame_counter_idle >= 1000:
            REWARD = - 10
            GAME_OVER = True
        return REWARD, GAME_OVER

    def move_it(self, action):
        #[RLUDS]
        self.move_tank(action)
        if action[4] == 1:
            self.execute_shoot(self.tank_yellow_shoot_allow)

    def restart(self):
        self.TY_mask.x = self.ty_pos_x
        self.TY_mask.y = self.ty_pos_y

class Tank_Green:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = [False, False, False, False]
        self.TG_face = TANK_GREEN_IMG[2]
        self.TG_face_txt = 'left'
        self.tank_green_shoot_allow = True
        self.tank_green_shoot_cooldown = False
        self.explosion_l_flag = False
        self.explosion_h_flag = False
        self.pos_init_find = True
        self.green_tank_destroyed = False
        self.green_tank_invicible = True
        self.frames_inv = 0
        self.bullet_dir = None

        self.eagle_greens_tank_on_hit_state = False
        self.yellow_tank_on_hit_state = False
        self.eagle_yellows_tank_on_hit_state = False

        self.AI_player = True
        self.Human_player = True

        for row in MAPPING:
            for col in row:
                if col == '2':
                    self.tg_pos_x = self.x
                    self.tg_pos_y = self.y
                self.x+=SQM
            self.y+=SQM
            self.x=0
        
        self.TG_mask = pygame.Rect(self.tg_pos_x, self.tg_pos_y, 52, 52)
     

    def bind(self, event):
        if event.type == KEYDOWN:
            if event.key == K_d:
                self.actions[0] = True
            elif event.key == K_a:
                self.actions[1] = True
            elif event.key == K_s:
                self.actions[2] = True
            elif event.key == K_w:
                self.actions[3] = True
        if event.type == KEYUP:
            if event.key == K_d:
                self.actions[0] = False
            elif event.key == K_a:
                self.actions[1] = False
            elif event.key == K_s:
                self.actions[2] = False
            elif event.key == K_w:
                self.actions[3] = False

    def move_tank(self, action):
        self.movement = [0,0]
        if action[0]:
            self.movement[0] += 8
            self.TG_face = TANK_GREEN_IMG[3]
            self.TG_face_txt = 'right'
        elif action[1]:
            self.movement[0] -= 8
            self.TG_face = TANK_GREEN_IMG[2]
            self.TG_face_txt = 'left'
        elif action[3]:
            self.movement[1] -= 8
            self.TG_face = TANK_GREEN_IMG[0]
            self.TG_face_txt = 'up'
        elif action[2]:
            self.movement[1] += 8
            self.TG_face = TANK_GREEN_IMG[1]
            self.TG_face_txt = 'down'
        self.TG_mask.x += self.movement[0]
        self.collisions_h = self.collision_test()
    
        for tile in self.collisions_h:
            if self.movement[0] > 0:
                self.TG_mask.right = tile.left
            if self.movement[0] < 0:
                self.TG_mask.left = tile.right
        
        self.TG_mask.y += self.movement[1]
        self.collisions_v = self.collision_test()
        
        for tile in self.collisions_v:
            if self.movement[1] > 0:
                self.TG_mask.bottom = tile.top
            if self.movement[1] < 0:
                self.TG_mask.top = tile.bottom

        self.collisions_sum = [self.collisions_h, self.collisions_v]
        
    def collision_test(self):
        colli = []
        for back in BACKGROUND_RECT:
            if self.TG_mask.colliderect(back):
                colli.append(back)
        for back in SOLID_RECT:
            if self.TG_mask.colliderect(back):
                colli.append(back)
        for back in BRICK_RECT:
            if self.TG_mask.colliderect(back):
                colli.append(back)
        for back in WATER_RECT:
            if self.TG_mask.colliderect(back):
                colli.append(back)
        for back in EAGLE_Y:
            if self.TG_mask.colliderect(back):
                colli.append(back)
        for back in EAGLE_G:
            if self.TG_mask.colliderect(back):
                colli.append(back)
        for back in BRICK_RECT_MINI:
            if self.TG_mask.colliderect(back):
                colli.append(back)
        return colli

    def bind_shoot(self, Flag):
        if Flag:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                flag_temp = True
                self.execute_shoot(flag_temp)

    def execute_shoot(self, Flag):
        if Flag:
            self.frames = 0
            self.tank_green_shoot_cooldown = True
            self.tank_green_shoot_allow = False
            self.b_tg = Bullet_TY(self.TG_mask.x, self.TG_mask.y, self.TG_face_txt)
            BULLETS_G_objects.append(self.b_tg)
            BULLETS_G_RECT.append(pygame.Rect(self.b_tg.x,self.b_tg.y,self.b_tg.width,self.b_tg.height))
            self.OHBG = On_Hit_By_Green(self.b_tg.dir)
            self.bullet_dir = self.b_tg.dir
              
    def shoot_delay(self, flag):
        if flag:
            if len(BULLETS_G_RECT) == 0 and self.frames > 20:
                self.tank_green_shoot_allow = True
                self.tank_green_shoot_cooldown = False
                self.bullet_dir = None
            self.frames += 1


    def bullets_onhit(self, TY_MASK, TG_CLASS, TY_CLASS, TY_DEST, TY_INVI, MAPPING,screen):
        if len(BULLETS_G_RECT) >= 1:
            for i, e in enumerate(BULLETS_G_RECT):
                self.explosion_l_flag = True
                self.explosion_h_flag = True
                self.brick_on_hit_state = self.OHBG.brick_on_hit(i, e)
                self.background_on_hit_state = self.OHBG.background_on_hit(i, e)
                self.yellow_tank_on_hit_state = self.OHBG.yellow_tank_on_hit(i, e, TY_MASK, TG_CLASS, TY_DEST, TY_INVI)
                self.solid_on_hit_state = self.OHBG.solid_on_hit(i, e)
                self.eagle_greens_tank_on_hit_state = self.OHBG.eagle_greens_tank_on_hit(i, e, TG_CLASS, TY_CLASS, MAPPING)
                self.eagle_yellows_tank_on_hit_state = self.OHBG.eagle_yellows_tank_on_hit(i, e, TG_CLASS, TY_CLASS, MAPPING)
                self.enemys_bullet_on_hit_state = self.OHBG.enemys_bullet_on_hit(i, e)
                self.states = [self.brick_on_hit_state, 
                self.background_on_hit_state,
                self.yellow_tank_on_hit_state,
                self.solid_on_hit_state,
                self.eagle_greens_tank_on_hit_state,
                self.eagle_yellows_tank_on_hit_state,
                self.enemys_bullet_on_hit_state]
                for xi in self.states:
                    if xi:
                        self.OHBG.break_bullet(i)

        if self.explosion_l_flag or self.explosion_h_flag:
            self.OHBG.draw_explosion_little(screen, self.explosion_l_flag)
            self.OHBG.draw_explosion_hard(screen, self.explosion_h_flag)

    def draw(self, screen, flag_1, flag_2):
        if flag_1 is False:
            screen.blit(self.TG_face,(self.TG_mask.x,self.TG_mask.y))
        if flag_2:
            if (self.frames_inv % 4) == 0 or (self.frames_inv % 4) == 1:
                screen.blit(INVICIBLE_1_IMG,(self.TG_mask.x,self.TG_mask.y))
            elif (self.frames_inv % 4) == 2 or (self.frames_inv % 4) == 3:
                screen.blit(INVICIBLE_2_IMG,(self.TG_mask.x,self.TG_mask.y))
            if self.frames_inv >= 45:
                self.green_tank_invicible = False
            self.frames_inv += 1

    def green_tank_position_relative_with_yellow_tank(self, TY_mask, TG_mask):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        if TG_mask.x <= TY_mask.x:
            flags[0] = True
        if TG_mask.x >= TY_mask.x:
            flags[1] = True
        if TG_mask.y >= TY_mask.y:
            flags[2] = True
        if TG_mask.y <= TY_mask.y:
            flags[3] = True
        return flags

    def green_eagle_position_relative_with_green_tank(self, TG_mask):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        for i in EAGLE_G:
            if TG_mask.x <= i.x:
                flags[0] = True
            if TG_mask.x >= i.x:
                flags[1] = True
            if TG_mask.y >= i.y:
                flags[2] = True
            if TG_mask.y <= i.y:
                flags[3] = True 
        return flags

    def yellow_eagle_position_relative_with_green_tank(self, TG_mask):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        for i in EAGLE_G:
            if TG_mask.x <= i.x:
                flags[0] = True
            if TG_mask.x >= i.x:
                flags[1] = True
            if TG_mask.y >= i.y:
                flags[2] = True
            if TG_mask.y <= i.y:
                flags[3] = True 
        return flags

    def green_tank_direction(self):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        if self.TG_face_txt == 'right':
            flags[0] = True
        elif self.TG_face_txt == 'left':
            flags[1] = True
        elif self.TG_face_txt == 'up':
            flags[2] = True
        elif self.TG_face_txt == 'down':
            flags[3] = True
        return flags

    def green_tank_bullet_presence(self):
        flag = False
        if self.tank_green_shoot_allow is True:
            flag = False
        elif self.tank_green_shoot_allow is False:
            flag = True
        return [flag]

    def green_tank_own_bullet_direction(self, dir, pres):
        #flags [R,L,U,D]
        flags = [False, False, False, False]
        if pres:
            if dir == 'right':
                flags[0] = True
            elif dir == 'left':
                flags[1] = True
            elif dir == 'up':
                flags[2] = True
            elif dir == 'down':
                flags[3] = True
        return flags

    def green_tank_faced_to_entity_solid(self, dir, TY_MASK, TG_MASK):
        self.xn = TG_MASK.x + 26
        self.yn = TG_MASK.y + 26
        if dir[0] is True:
            for i in range(44):
                self.xn += 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                self.loop_logic_background = self.green_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.green_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.green_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                #self.loop_logic_enemys_eagle = self.green_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                self.loop_logic_enemy = self.green_tank_faced_to_enemy_loop(self.sample, TY_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]
        if dir[1] is True:
            for i in range(44):
                self.xn -= 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                self.loop_logic_background = self.green_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.green_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.green_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                #self.loop_logic_enemys_eagle = self.green_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                self.loop_logic_enemy = self.green_tank_faced_to_enemy_loop(self.sample, TY_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]

        if dir[2] is True:
            for i in range(44):
                self.yn -= 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                self.loop_logic_background = self.green_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.green_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.green_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                #self.loop_logic_enemys_eagle = self.green_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                self.loop_logic_enemy = self.green_tank_faced_to_enemy_loop(self.sample, TY_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]


        if dir[3] is True:
            for i in range(44):
                self.yn += 16
                self.sample = pygame.Rect(self.xn,self.yn,1,1)
                self.loop_logic_background = self.green_tank_faced_to_entity_loop(self.sample, BACKGROUND_RECT)
                self.loop_logic_solid = self.green_tank_faced_to_entity_loop(self.sample, SOLID_RECT)
                #self.loop_logic_own_eagle= self.green_tank_faced_to_entity_loop(self.sample, EAGLE_G)
                #self.loop_logic_enemys_eagle = self.green_tank_faced_to_entity_loop(self.sample, EAGLE_Y)
                self.loop_logic_enemy = self.green_tank_faced_to_enemy_loop(self.sample, TY_MASK)
                self.logic_array = np.array([self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy])
                self.logic_array_single = np.where(self.logic_array == True)
                if len(self.logic_array_single[0]) >= 1: 
                    return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]
        return [self.loop_logic_background, self.loop_logic_solid, self.loop_logic_enemy]

    def green_tank_faced_to_entity_loop(self, sample, entity):
        self.sample = sample
        for ni in entity:
            if self.sample.colliderect(ni):
                return True
        return False

    def green_tank_faced_to_enemy_loop(self, sample, TY_MASK):
        self.sample = sample
        if self.sample.colliderect(TY_MASK):
            return True
        return False

    def green_tank_stuck(self, colli):
        if len(colli[0]) >= 1 or len(colli[1]) >= 1:
            return [True]
        return [False]

    def yellow_tank_got_hit(self, flag):
        if self.yellow_tank_on_hit_state:
            self.yellow_tank_on_hit_state = False
            print('Zielony czołg zniszczył Żółty czołg')
            return [True]
        else: 
            return [False]

    def green_eagle_got_hit_by_green(self, flag):
        if self.eagle_greens_tank_on_hit_state:
            self.eagle_greens_tank_on_hit_state = False
            print('Zielony czołg zniszczył swojego orła')
            return [True]
        else: return [False]

    def yellow_eagle_got_hit_by_green(self, flag):
        if self.eagle_yellows_tank_on_hit_state:
            self.eagle_yellows_tank_on_hit_state = False
            print('Zielony czołg zniszczył orła przeciwnika')
            return [False]
        else: 
            return [False]
    
    def green_tank_collision_sensor(self, TG_MASK):
        self.xs = TG_MASK.x - 2
        self.ys = TG_MASK.y - 2
        self.coli_sensor = pygame.Rect(self.xs,self.ys,56,56)
        for n in SOLID_RECT:
            if self.coli_sensor.colliderect(n):
                return [True]
        for n in WATER_RECT:
            if self.coli_sensor.colliderect(n):
                return [True]
        for n in BACKGROUND_RECT:
            if self.coli_sensor.colliderect(n):
                return [True]
        return [False]

    def play_step(self, action, yellow_tank_got_hit_by_green, green_tank_got_hit_by_yellow, green_eagle_got_hit_by_green, yellow_eagle_got_hit_by_green, green_tank_collision_sensor_state, frame_counter_idle):
        self.move_it(action)
        REWARD = 0
        GAME_OVER = False
        if green_tank_collision_sensor_state[0]:
            REWARD = - 0.1
        elif yellow_tank_got_hit_by_green[0]:
            GAME_OVER = True
            REWARD = 50
        elif green_tank_got_hit_by_yellow[0]:
            GAME_OVER = True
            REWARD = -50
        elif green_eagle_got_hit_by_green[0]:
            GAME_OVER = True
            REWARD = -150
        elif yellow_eagle_got_hit_by_green[0]:
            GAME_OVER = True
            REWARD = 150
        elif frame_counter_idle >= 1000:
            REWARD = - 10
            GAME_OVER = True
        return REWARD, GAME_OVER

    def move_it(self, action):
        #[RLUDS]
        self.move_tank(action)
        if action[4] == 1:
            self.execute_shoot(self.tank_green_shoot_allow)

    def restart(self):
        self.TG_mask.x = self.tg_pos_x
        self.TG_mask.y = self.tg_pos_y


class Main:
    def __init__(self):
        pygame.init()
        self.frame_counter = 0
        self.frame_counter_idle = 0
        self.window = pygame.display.set_mode((SQM*17,SQM*9))
        self.mapping = Mapping()
        self.ty = Tank_Yellow()
        self.tg = Tank_Green()
        self.AI_Y = AI_YELLOW()
        self.AI_G = AI_GREEN()
        self.clock = pygame.time.Clock()

    def runtime(self):
        self.run = True
        while self.run:
            self.window.fill((0,0,0))
            self.ty.move_tank(self.ty.actions)
            self.ty.draw(self.window, self.ty.yellow_tank_destroyed, self.ty.yellow_tank_invicible)
            self.ty.bind_shoot(self.ty.tank_yellow_shoot_allow)
            self.ty.shoot_delay(self.ty.tank_yellow_shoot_cooldown)

            self.tg.move_tank(self.tg.actions)
            self.tg.draw(self.window, self.tg.green_tank_destroyed, self.tg.green_tank_invicible)
            self.tg.bind_shoot(self.tg.tank_green_shoot_allow)
            self.tg.shoot_delay(self.tg.tank_green_shoot_cooldown)

            self.mapping.draw_props(self.window)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                self.ty.bind(event)
                self.tg.bind(event)

            for b_ty in BULLETS_Y_objects:
                b_ty.draw(self.window)
                b_ty.move()
            
            for i in BULLETS_Y_RECT:
                b_ty.movehitbox(i)

            for b_tg in BULLETS_G_objects:
                b_tg.draw(self.window)
                b_tg.move()
            
            for i in BULLETS_G_RECT:
                b_tg.movehitbox(i)

            self.ty.bullets_onhit(self.tg.TG_mask, self.tg, self.ty, self.tg.green_tank_destroyed, self.tg.green_tank_invicible, self.mapping, self.window)
            self.tg.bullets_onhit(self.ty.TY_mask, self.ty, self.tg, self.ty.yellow_tank_destroyed, self.ty.yellow_tank_invicible, self.mapping, self.window)

            #Generowanie state
            #Pozycje dwóch czołgów względem siebie - 4 State
            self.yellow_tank_position_relative_with_green_tank_state = self.ty.yellow_tank_position_relative_with_green_tank(self.ty.TY_mask, self.tg.TG_mask)
            self.green_tank_position_relative_with_yellow_tank_state = self.tg.green_tank_position_relative_with_yellow_tank(self.ty.TY_mask, self.tg.TG_mask)
            #Pozycja własnego orła względem czołgu - 4 State
            self.yellow_eagle_position_relative_with_yellow_tank_state = self.ty.yellow_eagle_position_relative_with_yellow_tank(self.ty.TY_mask)
            self.green_eagle_position_relative_with_green_tank_state = self.tg.green_eagle_position_relative_with_green_tank(self.tg.TG_mask)
            #Pozycja obcego orła względem czołgu - 4 State
            self.green_eagle_position_relative_with_yellow_tank_state = self.ty.green_eagle_position_relative_with_yellow_tank(self.ty.TY_mask)
            self.yellow_eagle_position_relative_with_green_tank_state = self.tg.yellow_eagle_position_relative_with_green_tank(self.tg.TG_mask)
            #Zwrot swojego czołgu - 4 State
            self.yellow_tank_direction_state = self.ty.yellow_tank_direction()
            self.green_tank_direction_state = self.ty.yellow_tank_direction()
            #Obecność swojego pocisku - 1 State
            self.yellow_tank_own_bullet_presence_state = self.ty.yellow_tank_bullet_presence()
            self.green_tank_own_bullet_presence_state = self.tg.green_tank_bullet_presence()
            #Obecność posicsku swojego przeciwnika - 1 State
            self.yellow_tank_enemys_bullet_presence_state = self.green_tank_own_bullet_presence_state
            self.green_tank_enemys_bullet_presence_state = self.yellow_tank_own_bullet_presence_state
            #Kierunek swojego pocisku - 4 State
            self.yellow_tank_own_bullet_direction_state = self.ty.yellow_tank_own_bullet_direction(self.ty.bullet_dir, self.yellow_tank_own_bullet_presence_state)
            self.green_tank_own_bullet_direction_state = self.tg.green_tank_own_bullet_direction(self.tg.bullet_dir, self.green_tank_own_bullet_presence_state)
            #Kierunek pocisku przeciwnika - 4 State
            self.yellow_tank_enemys_bullet_direction_state = self.green_tank_own_bullet_direction_state
            self.green_tank_enemys_bullet_direction_state = self.yellow_tank_own_bullet_direction_state
            #Kierunek zwrotu czołgu do obiektów - Background, Solid, Eagle_own, Eagle_enemy, Enamy_tank - 5 State
            #Wyłączono ją Tymaczasowo
            self.yellow_tank_faced_to_entity_solid_state = self.ty.yellow_tank_faced_to_entity_solid(self.yellow_tank_direction_state, self.ty.TY_mask, self.tg.TG_mask, self.window)
            self.green_tank_faced_to_entity_solid_state = self.tg.green_tank_faced_to_entity_solid(self.green_tank_direction_state, self.ty.TY_mask, self.tg.TG_mask)
            #Czy dany czołg utkną - 1 State
            #self.yellow_tank_stuck_state = self.ty.yellow_tank_stuck(self.ty.collisions_sum)
            #self.green_tank_stuck_state = self.tg.green_tank_stuck(self.tg.collisions_sum)
            #Czy czołg otrzymał obrażenia - 1 State
            self.green_tank_got_hit_by_yellow_state = self.ty.green_tank_got_hit(self.yellow_tank_own_bullet_presence_state)
            self.yellow_tank_got_hit_by_green_state = self.tg.yellow_tank_got_hit(self.green_tank_own_bullet_presence_state)
            #Czy orzeł swój otrzymał obrażenia - 1 State
            self.yellow_eagle_got_hit_by_yellow_state = self.ty.yellow_eagle_got_hit_by_yellow(self.yellow_tank_own_bullet_presence_state)
            self.green_eagle_got_hit_by_green_state = self.tg.green_eagle_got_hit_by_green(self.green_tank_own_bullet_presence_state)
            #Czy orzeł przeciwnika otrzymał obrażenia - 1 State
            self.green_eagle_got_hit_by_yellow_state = self.ty.green_eagle_got_hit_by_yellow(self.yellow_tank_own_bullet_presence_state)
            self.yellow_eagle_got_hit_by_green_state = self.tg.yellow_eagle_got_hit_by_green(self.green_tank_own_bullet_presence_state)
            #Sensor kolizyjny 1 State
            self.yellow_tank_collision_sensor_state = self.ty.yellow_tank_collision_sensor(self.ty.TY_mask)
            self.green_tank_collision_sensor_state = self.tg.green_tank_collision_sensor(self.tg.TG_mask)

            #Get State Yellow
            yellow_tank_current_state_old = self.AI_Y.get_state(
                self.yellow_tank_position_relative_with_green_tank_state, 
                #self.yellow_eagle_position_relative_with_yellow_tank_state,
                #self.green_eagle_position_relative_with_yellow_tank_state,
                self.yellow_tank_direction_state,
                self.yellow_tank_own_bullet_presence_state,
                self.yellow_tank_enemys_bullet_presence_state,
                self.yellow_tank_own_bullet_direction_state,
                self.yellow_tank_enemys_bullet_direction_state,
                self.yellow_tank_faced_to_entity_solid_state,
                self.yellow_tank_collision_sensor_state,
                self.green_tank_got_hit_by_yellow_state,
                self.yellow_tank_got_hit_by_green_state,
                #self.yellow_eagle_got_hit_by_yellow_state,
                #self.green_eagle_got_hit_by_yellow_state
            )
             
            move_calculated = self.AI_Y.get_action(yellow_tank_current_state_old, self.frame_counter)

            reward_y, done_y = self.ty.play_step(move_calculated, 
                self.green_tank_got_hit_by_yellow_state, 
                self.yellow_tank_got_hit_by_green_state, 
                self.yellow_eagle_got_hit_by_yellow_state,
                self.green_eagle_got_hit_by_yellow_state,
                self.yellow_tank_collision_sensor_state,
                self.frame_counter_idle
                )
        
            yellow_tank_current_state_new = self.AI_Y.get_state(
                self.yellow_tank_position_relative_with_green_tank_state, 
                #self.yellow_eagle_position_relative_with_yellow_tank_state,
                #self.green_eagle_position_relative_with_yellow_tank_state,
                self.yellow_tank_direction_state,
                self.yellow_tank_own_bullet_presence_state,
                self.yellow_tank_enemys_bullet_presence_state,
                self.yellow_tank_own_bullet_direction_state,
                self.yellow_tank_enemys_bullet_direction_state,
                self.yellow_tank_faced_to_entity_solid_state,
                self.yellow_tank_collision_sensor_state,
                self.green_tank_got_hit_by_yellow_state,
                self.yellow_tank_got_hit_by_green_state,
                #self.yellow_eagle_got_hit_by_yellow_state,
                #self.green_eagle_got_hit_by_yellow_state
            )
            self.AI_Y.train_short_memory(yellow_tank_current_state_old, move_calculated, reward_y, yellow_tank_current_state_new, done_y)
            self.AI_Y.remember(yellow_tank_current_state_old, move_calculated, reward_y, yellow_tank_current_state_new, done_y)
            final_score_value_y = self.AI_Y.final_score(reward_y)
            self.AI_Y.print_state(yellow_tank_current_state_old, self.frame_counter, final_score_value_y)
            
            #Get State Green

            green_tank_current_state_old = self.AI_G.get_state(
                self.green_tank_position_relative_with_yellow_tank_state, 
                #self.green_eagle_position_relative_with_green_tank_state,
                #self.yellow_eagle_position_relative_with_green_tank_state,
                self.green_tank_direction_state,
                self.green_tank_own_bullet_presence_state,
                self.green_tank_enemys_bullet_presence_state,
                self.green_tank_own_bullet_direction_state,
                self.green_tank_enemys_bullet_direction_state,
                self.green_tank_faced_to_entity_solid_state,
                self.green_tank_collision_sensor_state,
                self.yellow_tank_got_hit_by_green_state,
                self.green_tank_got_hit_by_yellow_state,
                #self.yellow_eagle_got_hit_by_yellow_state,
                #self.green_eagle_got_hit_by_yellow_state
            )
            
            move_calculated = self.AI_G.get_action(green_tank_current_state_old, self.frame_counter)

            reward_g, done_g = self.tg.play_step(move_calculated, 
                self.yellow_tank_got_hit_by_green_state, 
                self.green_tank_got_hit_by_yellow_state, 
                self.green_eagle_got_hit_by_green_state,
                self.yellow_eagle_got_hit_by_green_state,
                self.green_tank_collision_sensor_state,
                self.frame_counter_idle
                )
        
            green_tank_current_state_new = self.AI_G.get_state(
                self.green_tank_position_relative_with_yellow_tank_state, 
                #self.green_eagle_position_relative_with_green_tank_state,
                #self.yellow_eagle_position_relative_with_green_tank_state,
                self.green_tank_direction_state,
                self.green_tank_own_bullet_presence_state,
                self.green_tank_enemys_bullet_presence_state,
                self.green_tank_own_bullet_direction_state,
                self.green_tank_enemys_bullet_direction_state,
                self.green_tank_faced_to_entity_solid_state,
                self.green_tank_collision_sensor_state,
                self.yellow_tank_got_hit_by_green_state,
                self.green_tank_got_hit_by_yellow_state,
                #self.yellow_eagle_got_hit_by_yellow_state,
                #self.green_eagle_got_hit_by_yellow_state
            )
            self.AI_G.train_short_memory(green_tank_current_state_old, move_calculated, reward_g, green_tank_current_state_new, done_g)
            self.AI_G.remember(green_tank_current_state_old, move_calculated, reward_g, green_tank_current_state_new, done_g)
            final_score_value_g = self.AI_G.final_score(reward_g)
            self.AI_G.print_state(green_tank_current_state_old, self.frame_counter, final_score_value_g)

            if_done_state = self.if_done(done_g, done_y)
            if if_done_state:
                self.frame_counter_idle = 0
            self.restart_game(if_done_state)

            self.frame_counter += 1
            self.frame_counter_idle += 1
            pygame.display.update()
            self.clock.tick(FPS)

    def restart_game(self,if_done_state):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_p] or if_done_state:
            self.ty.restart()
            self.tg.restart()
    def if_done(self, dg, dy):
        if dg or dy:
            return True
        else: return False

if __name__ == '__main__':
    main = Main()
    main.runtime()