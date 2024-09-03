import argparse
import random
from pyamaze import maze, agent, COLOR
from Markov import Markov


def create_maze(size, loop_percent):
    """
    Create a maze of the given size with a specified loop percentage.

    Args:
        size (int): The size of the maze (MxM)
        loop_percent (int): The loop percentage for the maze.

    Returns:
        maze: An instance of the pyamaze maze class
    """
    my_maze = maze(size, size)
    my_maze.CreateMaze(loopPercent=loop_percent)

    return my_maze


def main():
    """
    Main function to run the MazeRunner program.
    Parses command line arguments for maze size and loop percentage, create the maze,
    initializes the Markov model, performs transitions, and visualizes the agent's paths.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(description="MazeRunner")
    parser.add_argument("size", type=int, choices=[10, 20], help="Size of the maze (MxM)")
    parser.add_argument("loopperc", type=int, choices=[0, 50], help="Loop percentage for the maze")

    # Parse arguments
    args = parser.parse_args()
    maze_size = args.size
    loop_percent = args.loopperc

    # Create the maze
    my_maze = create_maze(maze_size, loop_percent)

    # Initialize the Markov model
    markov = Markov(my_maze, maze_size)

    # Generate random start and goal states
    random_state = (random.randint(1, maze_size), random.randint(1, maze_size))
    goal_state = (random.randint(1, maze_size), random.randint(1, maze_size))

    # Print initial information
    print(f"\nMaze size {maze_size}x{maze_size} with loop percent {loop_percent}")
    print("Initial Transition Matrix:")
    print(markov.transition_matrix)

    # Perform Markov transitions and write results to readme.txt
    markov.markov_transition(random_state, goal_state)

    # Print start and goal states
    print(f"Random start state: {random_state}")
    print(f"Goal state: {goal_state}")

    # Find and prints paths
    paths = markov.show_paths(random_state, goal_state)

    # Print steady state matrix
    print("\nSteady State Distribution:")
    print(markov.steady_state_distribution())

    # Visualize paths in GUI
    agents = []
    colors = [COLOR.red, COLOR.cyan, COLOR.yellow]
    for i, path in enumerate(paths):
        my_agent = agent(my_maze, random_state[0], random_state[1], footprints=True, filled=True,
                         color=colors[i % len(colors)])
        my_maze.tracePath({my_agent: path})
        agents.append(my_agent)

    my_maze.run()


if __name__ == "__main__":
    main()
