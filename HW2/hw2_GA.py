import random
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, capacity, weights, profits, population_size, generations):
        """
        初始化基因演算法的參數。

        Parameters:
            capacity (int): 背包的容量。
            weights (list): 每個物件的權重。
            profits (list): 每個物件的利潤。
            population_size (int): 個體數量。
            generations (int): 世代數量。
        """
        self.capacity = capacity
        self.weights = weights
        self.profits = profits
        self.population_size = population_size
        self.generations = generations

    def generate_random_individual(self):
        """
        隨機生成一個個體。

        Returns:
            list: 表示個體的列表，每個元素代表是否選擇對應物件。
        """
        return [random.randint(0, 1) for _ in range(len(self.weights))]

    def calculate_fitness(self, individual):
        """
        計算個體的適應度。

        Parameters:
            individual (list): 表示個體的列表，每個元素代表是否選擇對應物件。

        Returns:
            int: 個體的適應度，即背包內物件的總利潤。
        """
        total_weight = sum(individual[i] * self.weights[i]
                           for i in range(len(self.weights)))
        total_profit = sum(individual[i] * self.profits[i]
                           for i in range(len(self.profits)))
        return total_profit if total_weight <= self.capacity else 0

    def generate_initial_population(self):
        """
        生成初始個體群體。

        Returns:
            list: 初始個體群體。
        """
        return [self.generate_random_individual() for _ in range(self.population_size)]

    def tournament_selection(self, population):
        """
        選擇個體。

        Parameters:
            population (list): 當前個體群體。

        Returns:
            list: 被選中的個體。
        """
        tournament_size = 3
        tournament_individuals = random.sample(population, tournament_size)
        return max(tournament_individuals, key=self.calculate_fitness)

    def crossover(self, parent1, parent2):
        """
        交叉操作。

        Parameters:
            parent1 (list): 第一個父母個體。
            parent2 (list): 第二個父母個體。

        Returns:
            list: 子代個體。
        """
        crossover_point = random.randint(0, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutation(self, individual):
        """
        突變操作。

        Parameters:
            individual (list): 待突變的個體。

        Returns:
            list: 突變後的個體。
        """
        mutation_point = random.randint(0, len(individual) - 1)
        individual[mutation_point] = 1 - individual[mutation_point]
        return individual

    def genetic_algorithm(self):
        """
        執行基因演算法。

        Returns:
            list: 收斂過程中每一世代的最佳適應度值。
        """
        population = self.generate_initial_population()
        convergence_values = [
            max([self.calculate_fitness(individual) for individual in population])]

        for _ in range(self.generations):
            new_population = []

            for _ in range(self.population_size):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                child = self.crossover(parent1, parent2)
                if random.random() < 0.1:  # Mutation rate
                    child = self.mutation(child)

                new_population.append(child)

            population = new_population
            convergence_values.append(
                max([self.calculate_fitness(individual) for individual in population]))

        return convergence_values


def plot_convergence(convergence_values):
    """
    繪製收斂圖。

    Parameters:
        convergence_values (list): 收斂過程中每次迭代的最佳適應度值。
    """
    plt.plot(range(len(convergence_values)), convergence_values)
    plt.xlabel('Generations')
    plt.ylabel('Convergence Value')
    plt.title('Convergence Plot')
    plt.show()


def main():
    capacity = 170
    weights = [41, 50, 49, 59, 55, 57, 60]
    profits = [442, 525, 511, 593, 546, 564, 617]
    population_size = 50
    generations = 100

    ga = GeneticAlgorithm(capacity, weights, profits,
                          population_size, generations)
    convergence_values = ga.genetic_algorithm()
    plot_convergence(convergence_values)


if __name__ == "__main__":
    main()
