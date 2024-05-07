from evolutionaryAlgorithm import *
from neuralNetwork import *
from minimax import *
from finetuning import *

ea = EA(population_size= POPULATION_SIZE, generations= GENERATIONS, mutation_rate= MUTATION_RATE, offsprings= OFFSPRINGS, replacement=0, elite=0)

# Initialize and run evolutionary algorithm
avg_BSF = [0 for _ in range(GENERATIONS)]
avg_ASF = [0 for _ in range(GENERATIONS)]
best_solutions = []

for iteration in range(1):
    # Initialize a new random population for each iteration
    population = ea.initialize_population()
    # print(population.individuals[0].input_size)
    best_fitness_values = []
    average_fitness_values = []
    
    population, best_individual, best_fit, best_fitness_values, average_fitness_values = ea.run(population)
    # best_solutions.append(best_individual)
    # avg_BSF = [x + y for x, y in zip(avg_BSF, best_fitness_values)]
    # avg_ASF = [x + y for x, y in zip(avg_ASF, average_fitness_values)]
    # print(f"Iteration: {iteration + 1}, Best Fitness: {best_fit}")

# Calculate average fitness over iterations
# avg_BSF = [x / 1 for x in avg_BSF]
# avg_ASF = [x / 1 for x in avg_ASF]

# # Plotting
# generations = range(1, len(best_fitness_values) + 1)

# plt.plot(generations, avg_BSF, label='Mean Best Fitness')
# plt.plot(generations, avg_ASF, label='Mean Average Fitness')
# plt.xlabel('Generations')
# plt.ylabel('Fitness')
# plt.title('Mean Best and Average Fitness over Iterations')
# plt.legend()
# plt.show()

# halloffame_filename = "HallofFame.txt"
# with open(halloffame_filename, 'a') as halloffame_file:
#     for i in best_solutions:
#         fitness_score = i.fitness(tsp_data)
#         if fitness_score <= 12000:
#             halloffame_file.write(f"{i.solution}; {i.fitness(tsp_data)}\n")