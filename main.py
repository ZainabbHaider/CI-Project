from evolutionaryAlgorithm import *
from neuralNetwork import *
from minimax import *
from finetuning import *
import matplotlib.pyplot as plt

ea = EA(population_size= POPULATION_SIZE, generations= GENERATIONS, mutation_rate= MUTATION_RATE, offsprings= OFFSPRINGS, replacement=0, elite=0)

# Initialize and run evolutionary algorithm
best_solutions = []

# Initialize a new random population for each iteration
population = ea.initialize_population()
best_fitness_values = []
average_fitness_values = []

population, best_individual, best_fit, best_fitness_values, average_fitness_values = ea.run(population)
best_solutions.append(best_individual)

# Plotting
generations = range(1, len(best_fitness_values) + 1)

plt.plot(generations, best_fitness_values, label='Best Fitness')
plt.plot(generations, average_fitness_values, label='Average Fitness')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Best and Average Fitness over Generations')
plt.legend()
plt.show()

# halloffame_filename = "HallofFame.txt"
# with open(halloffame_filename, 'a') as halloffame_file:
#     for i in best_solutions:
#         fitness_score = i.fitness(tsp_data)
#         if fitness_score <= 12000:
#             halloffame_file.write(f"{i.solution}; {i.fitness(tsp_data)}\n")