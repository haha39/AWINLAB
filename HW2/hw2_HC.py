import random


class HillClimbing:
    def __init__(self, capacity, weights, profits, iterations):
        self.capacity = capacity
        self.weights = weights
        self.profits = profits
        self.iterations = iterations

    def generate_random_solution(self):
        return [random.randint(0, 1) for _ in range(len(self.weights))]

    def calculate_fitness(self, solution):
        total_weight = sum(solution[i] * self.weights[i]
                           for i in range(len(self.weights)))
        total_profit = sum(solution[i] * self.profits[i]
                           for i in range(len(self.profits)))
        return total_profit if total_weight <= self.capacity else 0


def main():
    capacity = 170
    weights = [41, 50, 49, 59, 55, 57, 60]
    profits = [442, 525, 511, 593, 546, 564, 617]
    iterations = 100

    hc = HillClimbing(capacity, weights, profits, iterations)
    convergence_values = hc.hill_climbing()
    plot_convergence(convergence_values)


if __name__ == "__main__":
    main()
