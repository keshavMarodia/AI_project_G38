import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random

# Load the image
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding to convert the image to black and white
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define the PSO parameters
NUM_PARTICLES = 100
NUM_ITERATIONS = 50
W = 0.5
C1 = 2.0
C2 = 2.0

# Define the fitness function to be the number of white pixels in a particle's position
def fitness(position):
    x_min = min(x for x, y in position)
    x_max = max(x for x, y in position)
    y_min = min(y for x, y in position)
    y_max = max(y for x, y in position)
    roi = thresh[x_min:x_max+1, y_min:y_max+1]
    return np.count_nonzero(roi == 255)

# Define the Particle class
class Particle:
    def __init__(self):
        self.position = []
        self.velocity = []
        for i in range(4):
            x = random.randint(0, thresh.shape[0]-1)
            y = random.randint(0, thresh.shape[1]-1)
            self.position.append((x, y))
            self.velocity.append((0, 0))
        self.best_position = self.position
        self.best_fitness = fitness(self.position)

    def update_position(self):
        for i in range(4):
            x, y = self.position[i]
            vx, vy = self.velocity[i]
            rp, rg = random.random(), random.random()
            new_vx = W * vx + C1 * rp * (self.best_position[i][0] - x) + C2 * rg * (global_best[i][0] - x)
            new_vy = W * vy + C1 * rp * (self.best_position[i][1] - y) + C2 * rg * (global_best[i][1] - y)
            new_x = round(x + new_vx)
            new_y = round(y + new_vy)
            if new_x < 0:
                new_x = 0
            elif new_x >= thresh.shape[0]:
                new_x = thresh.shape[0] - 1
            if new_y < 0:
                new_y = 0
            elif new_y >= thresh.shape[1]:
                new_y = thresh.shape[1] - 1
            self.position[i] = (new_x, new_y)
            self.velocity[i] = (new_vx, new_vy)

        current_fitness = fitness(self.position)
        if current_fitness > self.best_fitness:
            self.best_position = self.position
            self.best_fitness = current_fitness

# Initialize the particles and global best
particles = [Particle() for _ in range(NUM_PARTICLES)]
global_best = particles[0].best_position
global_best_fitness = particles[0].best_fitness

# Run the PSO algorithm for the specified number of iterations
for i in range(NUM_ITERATIONS):
    for particle in particles:
        particle.update_position()
        if particle.best_fitness > global_best_fitness:
            global_best = particle.best_position
            global_best_fitness = particle.best_fitness