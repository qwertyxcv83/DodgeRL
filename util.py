import pygame
import os


WIDTH = 1200
HEIGHT = 675
PLAYERWIDTH = 64
OBSTACLEWIDTH = 30
PRESENTWIDTH = 100
PRESENTHEIGHT = 75
PLAYERSPEED = 0.3
OBSTACLESPEED = PLAYERSPEED
OBJECTSHOR = 5
OBJECTSVER = 7
ACT_DRAW_LENGTH = 60
ACT_GRAD_LENGTH = 10


def load_image(filename):
    return pygame.image.load(os.path.join("resources", filename))


def rect_collide(centerX1, centerY1, centerX2, centerY2, width1, height1, width2, height2):
    return abs(centerX1-centerX2) < (width1 + width2)/2 and abs(centerY1-centerY2) < (height1 + height2)/2


def is_in_rect(x, y, centerx, centery, width, height):
    return abs(x-centerx) < width/2 and abs(y-centery) < height/2


def toEncodeScaleX(x):
    return x / WIDTH * 2 - 1


def toEncodeScaleY(y):
    return y / HEIGHT * 2 - 1


def toDrawScaleX(x):
    return (x+1) / 2 * WIDTH


def toDrawScaleY(y):
    return (y+1) / 2 * HEIGHT


