# Bidirectional-Search-using-MM-and-MM0

We have implemented a Bi-diretional Serach using MM and MM0 algorithm, for a Pac-Man Domain, which is guaranteed to meet in the middle. We have provided a comparative stduy of MM, MM0, A*, BFS and DFS. These algorithms are tested in different environments, ranging in size, number of obstacles (complexity) and the initial positions of Pacman.

Our code is inspired from the paper, “Bidirectional Search That Is Guaranteed to Meet in the Middle”, Robert C. Holte, Ariel Felner, Guni Sharon, Nathan R. Sturtevant, AAAI 2016.

# What you will need to run:
* Python 2.7

# How to run:
Navigate to the search folder of Bidirectional_Search. To test the alogrithms on mazes of different sizes and complexities, enter the following commands:

## To check the results for 0% complexity:

* A* :- python pacman.py -l tinyMaze0 -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
* MM0 :- python pacman.py -l tinyMaze0 -z .5 -p SearchAgent -a fn=bdmm0,heuristic=manhattanHeuristic
* MM :- python pacman.py -l tinyMaze0 -z .5 -p SearchAgent -a fn=bdmm,heuristic=manhattanHeuristic
To check the complexity for medium and big maze, replace tinyMaze0 by mediumMaze0 or bigMaze0

## To check the results for 30% complexity:

* A* :- python pacman.py -l mediumMaze30 -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
* MM0 :- python pacman.py -l mediumMaze30 -z .5 -p SearchAgent -a fn=bdmm0,heuristic=manhattanHeuristic
* MM :- python pacman.py -l mediumMaze30 -z .5 -p SearchAgent -a fn=bdmm,heuristic=manhattanHeuristic

To check the complexity for tiny and big maze, replace mediumMaze30 by tinyMaze30 or bigMaze30

## To check the results for 50% complexity:
* A* :- python pacman.py -l bigMaze50 -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
* MM0 :- python pacman.py -l bigMaze50 -z .5 -p SearchAgent -a fn=bdmm0,heuristic=manhattanHeuristic
* MM :- python pacman.py -l bigMaze50 -z .5 -p SearchAgent -a fn=bdmm,heuristic=manhattanHeuristic
* BFS :- python pacman.py -l bigMaze50 -z .5 -p SearchAgent -a fn=bfs
* DFS :- python pacman.py -l bigMaze50 -z .5 -p SearchAgent -a fn=dfs

To check the complexity for medium maze, replace bigMaze50 by mediumMaze50
