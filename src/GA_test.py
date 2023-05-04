import cv2
import pytesseract
import numpy as np
from PIL import Image
import random

def ocr_ga(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Pre-processing the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(gray, 100, 200)

    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Define the fitness function for the GA
    def fitness(individual):
        config = f"--psm {individual[0]} --oem {individual[1]} -c tessedit_char_whitelist={individual[2]}"
        text = pytesseract.image_to_string(Image.fromarray(img_array))
        return sum([ord(c) for c in text if c.isalnum()])
    
    # Define the genetic operators
    def mutation(individual):
        i = random.randint(0, 2)
        if i == 0:
            individual[i] = random.randint(1, 13)
        elif i == 1:
            individual[i] = random.randint(1, 4)
        else:
            individual[i] = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 5))
    
    def crossover(parent1, parent2):
        child = parent1[:]
        i = random.randint(0, 2)
        child[i:] = parent2[i:]
        return child
    
    # Define the GA parameters
    population_size = 10
    generations = 20
    mutation_rate = 0.1
    
    # Initialize the population
    population = [[random.randint(1, 13), random.randint(1, 4), ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 5))] for _ in range(population_size)]
    
    # Evolve the population
    for i in range(generations):
        # Evaluate the fitness of each individual
        fitness_scores = [fitness(individual) for individual in population]
        
        # Select the parents for reproduction
        parents = [population[i] for i in np.argsort(fitness_scores)[-2:]]
        
        # Create the new generation
        new_population = parents[:]
        while len(new_population) < population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutation(child)
            new_population.append(child)
        
        # Update the population
        population = new_population
    
    # Select the best individual
    best_individual = max(population, key=fitness)
    
    # Perform OCR using the best individual
    config = f"--psm {best_individual[0]} --oem {best_individual[1]} -c tessedit_char_whitelist={best_individual[2]}"
    text = pytesseract.image_to_string(Image.fromarray(img_array))
    
    print(text)
    
    return text

ocr_ga("/home/abhijay/Desktop/Code/AI/github/AI_project_G38/src/line.png")
