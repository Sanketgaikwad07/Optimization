

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

  

class City:
  
    def __init__(self, name, lon, lat):
        self.name = name
        self.x = float(lon)
        self.y = float(lat)

    def distance(self, city):
        """Calculates the Euclidean distance to another city."""
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt(x_dis**2 + y_dis**2)
        return distance

    def __repr__(self):
        return f"{self.name}"


class Fitness:
    """
    Calculates the fitness of a route. Fitness is the inverse of the route distance.
    """
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
      #  self.__class__=0
        self.calculate_fitness()

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(len(self.route)):
                from_city = self.route[i]
                to_city = self.route[i-1]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance

    def calculate_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness

# --- 2. Geocoding Function ---

def get_cities_from_file(file_name):
    """
    Reads a list of city names from a file, fetches their coordinates using geopy,
    and returns a list of City objects.
    """
    city_list = []
    geolocator = Nominatim(user_agent="tsp_genetic_algorithm_solver")
    
    print("Fetching city coordinates... (This may take a moment)")
    
    try:
        with open(file_name, 'r') as f:
            for city_name in f:
                city_name = city_name.strip()
                if city_name:
                    try:
                        location = geolocator.geocode(city_name)
                        if location:
                            city = City(name=city_name, lon=location.longitude, lat=location.latitude)
                            city_list.append(city)
                            print(f"Successfully located: {city_name}")
                        else:
                            print(f"Warning: Could not locate '{city_name}'. Skipping.")
                        
                        # Respect Nominatim's usage policy (1 request per second)
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error geocoding {city_name}: {e}")
                        
    except FileNotFoundError:
        print(f"Error: The data file '{file_name}' was not found.")
        return None
        
    print("Finished fetching coordinates.\n")
    return city_list


# --- 3. Genetic Algorithm Components ---

def create_random_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route

def create_initial_population(pop_size, city_list):
    return [create_random_route(city_list) for _ in range(pop_size)]

def rank_routes(population):
    fitness_results = {i: Fitness(population[i]).fitness for i in range(len(population))}
    return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

def selection(pop_ranked, elite_size):
    selection_results = [pop_ranked[i][0] for i in range(elite_size)]
    for _ in range(len(pop_ranked) - elite_size):
        tournament = random.sample(pop_ranked, k=5)
        winner = max(tournament, key=lambda x: x[1])
        selection_results.append(winner[0])
    return selection_results

def mating_pool(population, selection_results):
    return [population[i] for i in selection_results]

def crossover(parent1, parent2):
    child = [None] * len(parent1)
    start_gene, end_gene = sorted(random.sample(range(len(parent1)), 2))
    child[start_gene:end_gene] = parent1[start_gene:end_gene]
    parent2_genes = [item for item in parent2 if item not in child]
    idx = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = parent2_genes[idx]
            idx += 1
    return child

def breed_population(matingpool, elite_size):
    children = matingpool[:elite_size]
    pool = random.sample(matingpool, len(matingpool))
    for i in range(len(matingpool) - elite_size):
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual

def mutate_population(population, mutation_rate):
    return [mutate(individual, mutation_rate) for individual in population]

def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    return mutate_population(children, mutation_rate)

# --- 4. Plotting Functions ---

def plot_route(route, generation_num, ax):
    ax.clear()
    x_coords = [city.x for city in route]
    y_coords = [city.y for city in route]
    x_coords.append(route[0].x)
    y_coords.append(route[0].y)
    
    ax.plot(x_coords, y_coords, 'o-', label='Route')
    for city in route:
        ax.text(city.x, city.y, f' {city.name.split(",")[0]}', fontsize=9)
        
    ax.set_title(f'Best Route - Generation {generation_num}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    plt.pause(0.05)

# --- 5. Main GA Execution ---

def genetic_algorithm(cities, pop_size, elite_size, mutation_rate, generations, plot_interval):
    population = create_initial_population(pop_size, cities)
    avg_fitness_progress = []

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    fig1.show()

    print("Starting Genetic Algorithm...")
    for i in range(generations):
        population = next_generation(population, elite_size, mutation_rate)
        
        current_fitness_sum = sum(Fitness(route).fitness for route in population)
        avg_fitness_progress.append(current_fitness_sum / len(population))
        
        ranked_pop = rank_routes(population)
        best_route_index = ranked_pop[0][0]
        best_route = population[best_route_index]
        best_distance = 1 / ranked_pop[0][1]
        
        print(f"Gen {i+1: >3}: Best Distance (approx) = {best_distance:.2f}, Avg Fitness = {avg_fitness_progress[-1]:.6f}")

        if (i + 1) % plot_interval == 0 or i == 0 or (i + 1) == generations:
            plot_route(best_route, i + 1, ax1)

    print("\nGenetic Algorithm finished.")
    final_best_distance = 1 / rank_routes(population)[0][1]
    print(f"Final best distance (approx): {final_best_distance:.2f}")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(avg_fitness_progress)
    ax2.set_title('Average Fitness vs. Generation')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Fitness')
    ax2.grid(True)
    plt.show(block=True)
    
    return final_best_distance, population[rank_routes(population)[0][0]]

if __name__ == '__main__':

    POPULATION_SIZE = 100
    ELITE_SIZE = 20
    MUTATION_RATE = 0.01
    GENERATIONS = 150
    PLOT_INTERVAL = 25
    
 
    DATA_FILE = 'India_cities_list.txt'
    
    city_list = get_cities_from_file(DATA_FILE)

    if city_list and len(city_list) > 1:
        best_dist, best_route_obj = genetic_algorithm(
            cities=city_list, 
            pop_size=POPULATION_SIZE, 
            elite_size=ELITE_SIZE, 
            mutation_rate=MUTATION_RATE, 
            generations=GENERATIONS,
            plot_interval=PLOT_INTERVAL
        )
        
        print("\nBest route found:")
        route_str = " -> ".join([city.name.split(',')[0] for city in best_route_obj])
        print(route_str + f" -> {best_route_obj[0].name.split(',')[0]}")
    elif city_list:
        print("Not enough cities to run the algorithm. Please provide at least 2 cities.")
