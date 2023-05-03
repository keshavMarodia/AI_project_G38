import numpy as np
import PIL.Image
import pytesseract
import random

# Define the input image filename and population size
input_image_filename = r'line4.jpeg'
population_size = 50

# Load the input image as a NumPy array
input_image = np.array(PIL.Image.open(input_image_filename))

# Define the OCR fitness function
def fitness(individual):
    ocr_text = pytesseract.image_to_string(PIL.Image.fromarray(individual))
    fitness_score = sum([1 for c in ocr_text if c.isalpha() or c.isdigit() or c.isspace()])/len(ocr_text)
    return fitness_score

# Define the genetic algorithm functions
def generate_individual():
    return np.random.randint(low=0, high=256, size=input_image.shape, dtype=np.uint8)

def crossover(parent1, parent2):
    child = np.zeros_like(parent1)
    mask = np.random.randint(low=0, high=2, size=parent1.shape, dtype=bool)
    child[mask] = parent1[mask]
    child[~mask] = parent2[~mask]
    return child

def mutate(individual):
    mutation_rate = 0.1
    mutation_mask = np.random.random(size=individual.shape) < mutation_rate
    mutation = np.random.randint(low=-50, high=50, size=individual.shape, dtype=np.int8)
    individual[mutation_mask] = np.clip(individual[mutation_mask] + mutation[mutation_mask], 0, 255)
    return individual

# Generate the initial population
population = [generate_individual() for _ in range(population_size)]

# Run the genetic algorithm for 100 generations
for generation in range(100):
    # Evaluate the fitness of each individual in the population
    fitness_scores = [fitness(individual) for individual in population]
    
    # Select the top 10% of the population as parents
    num_parents = int(population_size * 0.1)
    parents_indices = np.argsort(fitness_scores)[-num_parents:]
    parents = [population[i] for i in parents_indices]
    
    # Generate the next generation using crossover and mutation
    offspring = []
    while len(offspring) < population_size - num_parents:
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        offspring.append(child)
    population = parents + offspring

# Evaluate the fitness of the final population and select the best individual
fitness_scores = [fitness(individual) for individual in population]
best_individual_index = np.argmax(fitness_scores)
best_individual = population[best_individual_index]

# Perform OCR on the best individual and print the result
ocr_text = pytesseract.image_to_string(PIL.Image.fromarray(best_individual))
print("OCR result: ", ocr_text)