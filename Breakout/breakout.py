import pygame
from pygame.locals import *
import sys

class Breakout:
    def __init__(self):
        self.screen = pygame.display.set_mode((800, 600))
        self.blocks = []
        self.startingPaddleLocations = [300, 320, 340, 360]
        self.paddle = [[pygame.Rect(self.startingPaddleLocations[0], 500, 20, 10), 120],
                [pygame.Rect(self.startingPaddleLocations[1], 500, 20, 10),100],
                [pygame.Rect(self.startingPaddleLocations[2], 500, 20, 10),80],
                [pygame.Rect(self.startingPaddleLocations[3], 500, 20, 10),45],
        ]
        self.ball = pygame.Rect(300, 490, 5, 5)
        self.direction = -1
        self.yDirection = -1
        self.angle = 80

        # changed speeds here, when increasing levels increase this speed and examine preformance of agent
        # original were:
        # self.speeds = {
        #     120:(-10, -3),
        #     100:(-10, -8),
        #     80:(10, -8),
        #     45:(10, -3),
        # }
        self.speeds = {
            120:(-4, -3),
            100:(-4, -4),
            80:(4, -4),
            45:(4, -3),
        }
        self.swap = {
            120:45,
            45:120,
            100:80,
            80:100,
        }
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 25)
        self.score = 0

        # rl environment variables
        self.rewards = 0
        self.done = False

    def createBlocks(self):
        self.blocks = []
        y = 50
        for __ in range(int(200 / 10)):
            x = 50
            for _ in range(int(800 / 25 - 6)):
                block = pygame.Rect(x, y, 25, 10)
                self.blocks.append(block)
                x += 27
            y += 12

    def ballUpdate(self):
        for _ in range(2):
            speed = self.speeds[self.angle]
            xMovement = True
            if _:
                self.ball.x += speed[0] * self.direction
            else:
                self.ball.y += speed[1] * self.direction * self.yDirection
                xMovement = False
            if self.ball.x <= 0 or self.ball.x >= 800:
                self.angle = self.swap[self.angle]
                if self.ball.x <= 0:
                    self.ball.x = 1
                else:
                    self.ball.x = 799
            if self.ball.y <= 0:
                self.ball.y = 1
                self.yDirection *= -1

            ballHitPaddle = False
            for paddle in self.paddle:
                if paddle[0].colliderect(self.ball):
                    self.angle = paddle[1]
                    self.direction = -1
                    self.yDirection = -1

                    # ball hit paddle
                    ballHitPaddle = True
                    break

            if ballHitPaddle:
                self.rewards += 3


            check = self.ball.collidelist(self.blocks)
            if check != -1:
                # ball hits brick
                print("brick hit")
                self.rewards += 10

                block = self.blocks.pop(check)
                if xMovement:
                    self.direction *= -1
                self.yDirection *= -1
                self.score += 1

            if self.score == len(self.blocks):
                print("all blocks destroyed - you won")
                self.rewards += 800
                self.done = True

            if self.ball.y > 600:
                # ball misses paddle
                print("missed paddle")
                self.rewards -= 3
                self.done = True

                self.reset()

    # actions
    def goLeft(self):
        vx=-10
        self.updatePaddleLocation(vx)
        self.rewards -= 0.2

    def goRight(self):
        vx=10
        self.updatePaddleLocation(vx)
        self.rewards -= 0.2

    def updatePaddleLocation(self, move_x):
        on = 0
        # check if paddle is in dimension of the screen before updating
        if move_x + self.paddle[0][0].x > -5 and move_x + self.paddle[-1][0].x < self.screen.get_width():
            for p in self.paddle:
                p[0].x = p[0].x + move_x
                on += 1

    def paddleUpdate(self):
        keys=pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.goLeft()
        if keys[pygame.K_RIGHT]:
            self.goRight()

    # return array representing game state
    def getCurrentState(self):
        # state = [paddle x-location, ball x-location, ball y-location,
        #           ball x-speed, ball y-speed, bricks left]
        #
        # not using current score in state because it correlates with  bricks left
        # ie. bricks_left = total_bricks - score

        # using x-location of middle paddle
        paddleLocation = self.paddle[1][0].x
        ballXLocation = self.ball.x
        ballYLocation = self.ball.y
        speed = self.speeds[self.angle]
        ballXSpeed = speed[0]
        ballYSpeed = speed[1]
        bricksLeft = len(self.blocks) - self.score

        return [paddleLocation, ballXLocation, ballYLocation, ballXSpeed, ballYSpeed, bricksLeft]

    # one step in the environment
    # return state, rewards, done
    def step(self, action):
        self.rewards = 0
        self.done = False

        if action == 0:
            self.goLeft()
        elif action == 1:
            # go right
            self.goRight()
        elif action == 2:
            # do nothing
            pass

        self.ballUpdate()

        currentState = self.getCurrentState()

        return currentState, self.rewards, self.done

    def resetPaddleLocation(self):
        for i in range(len(self.paddle)):
            paddle = self.paddle[i][0]
            paddle.x = self.startingPaddleLocations[i]

    def reset(self):
        self.createBlocks()
        self.resetPaddleLocation()
        self.score = 0
        self.ball.x = self.paddle[1][0].x
        self.ball.y = 490
        self.yDirection = self.direction = -1

        currentState = self.getCurrentState()

        return currentState


    def main(self):
        pygame.mouse.set_visible(False)
        clock = pygame.time.Clock()
        self.createBlocks()
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
            self.screen.fill((0, 0, 0))
            self.paddleUpdate()
            self.ballUpdate()

            for block in self.blocks:
                pygame.draw.rect(self.screen, (255,255,255), block)
            for paddle in self.paddle:
                pygame.draw.rect(self.screen, (255,255,255), paddle[0])
            pygame.draw.rect(self.screen, (255,255,255), self.ball)
            self.screen.blit(self.font.render(str(self.score), -1, (255,255,255)), (400, 550))
            pygame.display.update()


if __name__ == "__main__":
    Breakout().main()
