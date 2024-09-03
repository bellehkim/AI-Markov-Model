import copy
from collections import deque
import numpy as numpy


class Markov:
    def __init__(self, maze, size):
        """
        Initializes the Markov model with a given maze and size.

        Args:
            maze (maze): The maze object.
            size (int): The size of the maze.
        """
        self.maze = maze
        self.size = size
        self.transition_matrix = self.create_transition_matrix()
        self.start = None
        self.goal = None

    def get_possible_moves(self, positions):
        """
        Gets possible moves from a given position in the maze.

        Args:
            positions (tuple): The current position (row, column).

        Returns:
            list: A list of possible moves from the current position.
        """
        curr_row, curr_col = positions
        direction_map = {
            'E': (0, 1),
            'W': (0, -1),
            'N': (-1, 0),
            'S': (1, 0)
        }
        result = []

        possible_directions = dict(self.maze.maze_map.get((curr_row, curr_col)))
        for direction, isOpen in possible_directions.items():
            if isOpen == 1:
                row_to_add, col_to_add = direction_map.get(direction)
                new_row = curr_row + row_to_add
                new_col = curr_col + col_to_add
                if 1 <= new_row <= self.maze.rows and 1 <= new_col <= self.maze.cols:
                    result.append((new_row, new_col))
        return result

    def create_transition_matrix(self):
        """
        Creates the transition matrix for the maze.

        Returns:
            numpy.ndarray: The transition matrix.
        """
        matrix = numpy.zeros((self.size * self.size, self.size * self.size))

        for row in range(1, self.size + 1):
            for col in range(1, self.size + 1):
                current_cell = (row - 1) * self.size + (col - 1)
                neighbors = self.get_possible_moves((row, col))

                if neighbors:
                    probability = 1 / len(neighbors)
                    for current_neighbor in neighbors:
                        neighbor_cell = (current_neighbor[0] - 1) * self.size + (current_neighbor[1] - 1)
                        matrix[current_cell, neighbor_cell] = probability

        return matrix

    def markov_transition(self, start_state, goal_state, max_steps=100):
        """
        Performs Markov transition to determine the number of steps to reach  the goal state.

        Args:
            start_state (tuple): The starting position (row, column).
            goal_state (tuple): The goal position (row, column).
            max_steps (int): The maximum number of steps to simulate.

        Writes:
            readme.txt: Outputs the transition matrices and steps.
        """
        matrix_row = (start_state[0] - 1) * self.size + (goal_state[1] - 1)
        matrix_col = (goal_state[0] - 1) * self.size + (goal_state[1])
        steps = 1
        current_distribution = self.transition_matrix.copy()
        paths = self.bfs(start_state, goal_state)
        solution_matrix = []

        with open("readme.txt", "w") as f:
            f.write(f"Maze Size: {self.size}x{self.size}\n")
            f.write(f"Initial Transition Matrix:\n{self.transition_matrix}\n\n")

            while steps <= max_steps:
                current_distribution = numpy.matmul(current_distribution, self.transition_matrix)
                steps += 1
                if len(paths) == 0:
                    print("Path not found")
                if steps % 10 == 0:
                    f.write(f"Steps: {steps}\n{current_distribution}\n\n")
                    print(f"Steps: {steps}: {current_distribution}")
                if steps >= max_steps:
                    break
                if len(paths) >= 1 and steps == len(paths[0]) - 1:
                    solution_matrix.append(copy.deepcopy(current_distribution))
                if len(paths) >= 2 and steps == len(paths[1]) - 1:
                    solution_matrix.append(copy.deepcopy(current_distribution))
                if len(paths) == 3 and steps == len(paths[2]) - 1:
                    solution_matrix.append(copy.deepcopy(current_distribution))

            f.write(f"Goal state reached in {steps} steps.\n")
            f.write(f"Transition matrix at goal state:\n"
                    f"{current_distribution.reshape(self.size * self.size, self.size * self.size)}\n")

            print(f"Goal state reached in {steps} steps.")
            print("Transition matrix at goal state:")
            print(current_distribution.reshape(self.size * self.size, self.size * self.size))

            for i, path in enumerate(solution_matrix):
                f.write(f"\nTransition matrix when agent reaches the goal state:\n"
                        f"Steps: {len(paths[i]) - 1}:\n{solution_matrix[i]}\n")
                print(f"\nTransition matrix when agent reaches the goal state:\n"
                      f"Steps: {len(paths[i]) - 1}:\n{solution_matrix[i]}\n")

            f.write(f"\nSteady State Distribution:\n{self.steady_state_distribution()}")

    def steady_state_distribution(self):
        """
        Calculate the steady state distribution matrix.

        Returns:
            numpy.ndarray: A array representing the steady state distribution matrix.

        Note:
            The function assumes that the transition matrix is a valid stochastic matrix
            where each row sums to 1.
        """
        eigvals, eigvecs = numpy.linalg.eig(self.transition_matrix.T)
        eigval_index_1 = numpy.argmax(numpy.isclose(eigvals, 1))
        steady_state_vec = numpy.real(eigvecs[:, eigval_index_1])
        steady_state_vec = steady_state_vec / numpy.sum(steady_state_vec)
        return steady_state_vec

    def bfs(self, start, goal, max_paths=3, max_steps=100):
        """
        Performs a BFS to find paths from start to goal state.

        Args:
            start (tuple): The starting position (row, column).
            goal (tuple): The goal position (row, column).
            max_paths (int): The maximum number of paths to find.
            max_steps (int): The maximum number of steps to search.

        Returns:
            list: A list of paths from start to goal.

        Note:
            The function assumes that a cell can be re-visited.
        """
        self.start = start
        self.goal = goal
        queue = deque([(self.start, [self.start], 0)])
        visited = set()
        paths = []

        while queue:
            current, path, steps = queue.popleft()
            if current == self.goal:
                paths.append(path)
                if len(paths) == max_paths:
                    return paths
            if steps < max_steps:
                for neighbor in self.get_possible_moves((current[0], current[1])):
                    # if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], steps + 1))
                # visited.add(current)

        return paths

    def show_paths(self, start_state, goal_state, max_paths=3):
        """
        Finds the prints path from start to goal state.

        Args:
            start_state (tuple): The starting position (row, column).
            goal_state (tuple): The goal position (row, column).
            max_paths (int): The maximum number of paths to find.

        Returns:
            list: A list of paths from start to goal.
        """
        paths = self.bfs(start_state, goal_state, max_paths)

        for i, path in enumerate(paths):
            formatted_path = ", ".join([f"({cell[0]},{cell[1]})" for cell in path])
            print(f"Path {i + 1}: {formatted_path}")

        return paths
