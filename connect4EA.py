import random
import ast
import copy
import matplotlib as plt
from main import *

def flatten(weights):
        flattened_weights = list(np.concatenate([arr.flatten() for arr in weights]))
        return flattened_weights

def unflatten(flattened_list, original_weights):
    unflattened_list = []
    start = 0
    for weights in original_weights:
        shape = weights.shape
        size = np.prod(shape)
        unflattened_list.append(np.array(flattened_list[start:start+size]).reshape(shape))
        start += size
    return unflattened_list

class Individual:
    def __init__(self, weights, biases, input_size, hidden_layers_sizes, output_size):
        weights = weights
        biases = biases
        input_size = input_size
        hidden_layers_sizes = hidden_layers_sizes
        output_size = output_size

    def fitness(self):
        nn = NeuralNetwork(self.input_size, self.hidden_layers_sizes, self.output_size, self.weights, self.biases)
        result = []
        for j in range(num_games):
            result.append(play_game(nn))
        return sum(result)

class Population:
    def __init__(self, individuals):
        super().__init__(individuals)

    def fitness_scores(self):
        return [individual.fitness() for individual in self.individuals]

    def crossover(self, parent1, parent2):
        parent1 = flatten(parent1.weights)
        parent2 = flatten(parent2.weights)
        # Child 1
        point1, point2 = sorted(random.sample(range(1, len(parent1.solution)), 2))
        child1 = []
        child1_middle = parent2.solution[point1:point2]  # Middle part from parent2
        remaining_num = []
        for i in range(point2, len(parent1.solution)):
            if parent1.solution[i] not in child1_middle:
                remaining_num.append(parent1.solution[i])
        for i in range(point2):
            if parent1.solution[i] not in child1_middle:
                remaining_num.append(parent1.solution[i])
        for i in range(point2, len(parent1.solution)):
            child1_middle.append(remaining_num.pop(0))
        
        for i in range(point1):
            child1.append(remaining_num.pop(0))
        child1 = child1 + child1_middle
        c1 = Individual(unflatten(child1, parent1.weights), parent1.biases, parent1.input_size, parent1.hidden_layers_sizes, parent1.output_size)
        
        # Child 2
        child2 = []
        child2_middle = parent1.solution[point1:point2]  # Middle part from parent1
        remaining_num2 = []
        for i in range(point2, len(parent2.solution)):
            if parent2.solution[i] not in child2_middle:
                remaining_num2.append(parent2.solution[i])
        for i in range(point2):
            if parent2.solution[i] not in child2_middle:
                remaining_num2.append(parent2.solution[i])
        for i in range(point2, len(parent2.solution)):
            child2_middle.append(remaining_num2.pop(0))
        for i in range(point1):
            child2.append(remaining_num2.pop(0))
        child2 = child2 + child2_middle    
        c2 = Individual(unflatten(child2, parent1.weights), parent1.biases, parent1.input_size, parent1.hidden_layers_sizes, parent1.output_size)
        
        return c1, c2

    # Perform mutation
    def mutate(self, s):
        solution = flatten(s.weights)
        mutated_solution = solution.solution[:]
        random_index_1 = random.randint(0, len(mutated_solution)-1)
        random_index_2 = random.randint(0, len(mutated_solution)-1)
        while random_index_1 == random_index_2:
            random_index_2 = random.randint(0, len(mutated_solution)-1)
            
        if random_index_1 > random_index_2:
            removed_item = mutated_solution.pop(random_index_1)
            mutated_solution.insert((random_index_2+1), removed_item)
        else:
            removed_item = mutated_solution.pop(random_index_2)
            mutated_solution.insert((random_index_1+1), removed_item)

        m = Individual(unflatten(mutated_solution, s.weights), s.biases, s.input_size, s.hidden_layers_sizes, s.output_size)
        return m

class EA:
    def __init__(self, population_size, generations, mutation_rate, offsprings, replacement, elite):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.offsprings = offsprings
        self.replacement = replacement
        self.elite = elite
        
    def random_selection(self, fitness_scores):
        random_number = random.randint(0, len(fitness_scores)-1)
        return random_number

    def truncation_selection_max(self, fitness_scores):
        max_value = max(fitness_scores)
        max_index = fitness_scores.index(max_value)
        return max_index

    def truncation_selection_min(self, fitness_scores):
        min_index = min(fitness_scores)
        min_index = fitness_scores.index(min_index)
        return min_index

    def fitness_proportional_selection_max(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        selected_index = random.choices(range(len(fitness_scores)), weights=probabilities)
        return selected_index[0]

    def fitness_proportional_selection_min(self, fitness_scores):
        inverted_fitness_scores = [1 / fitness for fitness in fitness_scores]
        total_inverted_fitness = sum(inverted_fitness_scores)
        probabilities = [inverted_fitness / total_inverted_fitness for inverted_fitness in inverted_fitness_scores]
        selected_index = random.choices(range(len(fitness_scores)), weights=probabilities)
        return selected_index[0]
    
    def rank_based_selection_max(self, fitness_scores):
        duplicate = copy.deepcopy(fitness_scores)
        dic = {}
        i = len(duplicate)
        while duplicate:
            m = max(duplicate)
            duplicate.remove(m)
            dic[m] = i
            i -= 1
        ranked = []
        for j in range(len(fitness_scores)):
            ranked.append(dic[fitness_scores[j]])
        return self.fitness_proportional_selection_max(ranked)

    def rank_based_selection_min(self, fitness_scores):
        duplicate = copy.deepcopy(fitness_scores)
        dic = {}
        i = len(duplicate)
        while duplicate:
            m = min(duplicate)
            duplicate.remove(m)
            dic[m] = i
            i -= 1
        ranked = []
        for j in range(len(fitness_scores)):
            ranked.append(dic[fitness_scores[j]])
        return self.fitness_proportional_selection_min(ranked)

    def binary_tournament_selection_max(self, fitness_scores):
        r1 = self.random_selection(fitness_scores)
        r2 = self.random_selection(fitness_scores)
        while r1 == r2:
            r2 = self.random_selection(fitness_scores)
        if fitness_scores[r1] > fitness_scores[r2]:
            return r1
        else:
            return r2
    
    def binary_tournament_selection_min(self, fitness_scores):
        r1 = self.random_selection(fitness_scores)
        r2 = self.random_selection(fitness_scores)
        while r1 == r2:
            r2 = self.random_selection(fitness_scores)
        if fitness_scores[r1] < fitness_scores[r2]:
            return r1
        else:
            return r2

    def initialize_population(self):
        individuals = []
        for i in range(self.population_size):
            weights, biases = initialise_weights(input_size, hidden_layers_sizes, output_size)
            individual = Individual(weights, biases, input_size, hidden_layers_sizes, output_size)
            individuals.append(individual)
        return Population(individuals)

    def run(self, tsp_data, pop):
        best_fitness_values = []
        avg_fitness_values = []
        
        for generation in range(self.generations):
            fitness_scores = pop.fitness_scores()

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(self.offsprings // 2):
                parent1 = pop.individuals[self.random_selection(fitness_scores)]
                parent2 = pop.individuals[self.random_selection(fitness_scores)]
                child1, child2 = pop.crossover(parent1, parent2)
                random_number_1 = random.random()
                random_number_2 = random.random()
                if random_number_1 > self.mutation_rate:
                    offspring.append(child1)
                else:
                    child1 = pop.mutate2(child1)
                    offspring.append(child1)
                if random_number_2 > self.mutation_rate:
                    offspring.append(child2)
                else:
                    child2 = pop.mutate2(child2)
                    offspring.append(child2)
            
            if self.replacement:
                # replacement implementation
                pop.individuals = offspring
            else:
                # regular implementation
                for i in offspring:
                    pop.individuals.append(i)
            
            if self.elite:
                file_path = "HallofFame.txt"
                elite_data = []
                with open(file_path, 'r') as file:
                    for line in file:
                        line = line.strip()  # Remove leading/trailing whitespaces and newline characters
                        if line:  # Ensure the line is not empty
                            items = line.split(';')  # Split the line by ';'
                            sol = ast.literal_eval(items[0])
                            sol = Individual(sol)
                            elite_data.append(sol)  # Append the extracted information as a tuple
                
                # add elite individuals to population
                r = random.random()
                if r < 0.7:
                    for i in elite_data:
                        if i not in pop.individuals:
                            r2 = random.random()
                            if r2 < 0.5:
                                pop.individuals.append(i)

            fitness_scores = pop.fitness_scores(tsp_data)
            
            temp_population = []
            for i in range(self.population_size):
                x = self.truncation_selection_max(fitness_scores)
                y = pop.individuals[x]
                pop.individuals.pop(x)
                fitness_scores.pop(x)
                temp_population.append(y)
            pop.individuals = temp_population
            fitness_scores = pop.fitness_scores(tsp_data)
                
            best_solution = min(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            
            best_fitness_values.append(best_solution)
            avg_fitness_values.append(average_fitness)
            
            print("Generation", generation+1, ": Best:",best_solution, "Average:", average_fitness)

        best_solution = min(fitness_scores)
        return pop, pop.individuals[fitness_scores.index(best_solution)], best_solution, best_fitness_values, avg_fitness_values
    

# run file
POPULATION_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.5
OFFSPRINGS = 20

ea = EA(population_size= POPULATION_SIZE, generations= GENERATIONS, mutation_rate= MUTATION_RATE, offsprings= OFFSPRINGS, replacement=0, elite=0)

# Initialize and run evolutionary algorithm
avg_BSF = [0 for _ in range(GENERATIONS)]
avg_ASF = [0 for _ in range(GENERATIONS)]
best_solutions = []

for iteration in range(1):
    # Initialize a new random population for each iteration
    population = ea.initialize_population()
    best_fitness_values = []
    average_fitness_values = []
    
    population, best_individual, best_fit, best_fitness_values, average_fitness_values = ea.run(tsp_data, population)
    best_solutions.append(best_individual)
    avg_BSF = [x + y for x, y in zip(avg_BSF, best_fitness_values)]
    avg_ASF = [x + y for x, y in zip(avg_ASF, average_fitness_values)]
    print(f"Iteration: {iteration + 1}, Best Fitness: {best_fit}")

# Calculate average fitness over iterations
avg_BSF = [x / 1 for x in avg_BSF]
avg_ASF = [x / 1 for x in avg_ASF]

# Plotting
generations = range(1, len(best_fitness_values) + 1)

plt.plot(generations, avg_BSF, label='Mean Best Fitness')
plt.plot(generations, avg_ASF, label='Mean Average Fitness')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Mean Best and Average Fitness over Iterations')
plt.legend()
plt.show()

# halloffame_filename = "HallofFame.txt"
# with open(halloffame_filename, 'a') as halloffame_file:
#     for i in best_solutions:
#         fitness_score = i.fitness(tsp_data)
#         if fitness_score <= 12000:
#             halloffame_file.write(f"{i.solution}; {i.fitness(tsp_data)}\n")