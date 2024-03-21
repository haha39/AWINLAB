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

    def genetic_algorithm(self):
        """
        執行基因演算法。

        Returns:
            list: 收斂過程中每一世代的最佳適應度值。
        """


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
    # plot_convergence(convergence_values)


if __name__ == "__main__":
    main()
