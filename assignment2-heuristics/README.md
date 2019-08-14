# Vehicle Routing Problem with Heuristics

## Problem
A supermarket chain is being acquired by a consorsium of other chains. A manager has a task to supervise the migration and required to visit every store as efficiently as possible. There is only 1 manager:
- every day has to start and end at the HQ
- average driving speed is 80 km/h
- distance is calculated with haversine formula
- visiting time is 1.5 hours for Jumbo and 1 hour otherwise
- the manager works at most 10 hours a day
- every store can be visited from 9:00h to 17:00h (only 8 hours)

The task is to find the route that minimize the total distance travelled by the manager, using, in order:
1. Nearest Neighbor algorithm to construct the initial solution
2. Two-edge exchange to improve the initial solution.
3. Tabu-search and simulated annealing to optimize the whole routes.
4. Ant-colony optimization to find the shortest path for each route

## Solution
1. input 	: contains the store list with their respective latitude-longitude as the input
2. results 	: contains the result of all the scripts as the solutions for the assignment
3. program 	: contains the python scripts to produce the results
	- model.py 					: the data structure representation, 
								imported by the other scripts
	- nearest_neighbors.py 		: nearest neighbors implementation
	- two_edge_exchange.py 		: 2-edge exchange implementation
	- tabu_search.py 			: tabu search implementation
	- simulated_annealing.py 	: simulated annealing implementation
	- ant_colony_opt.py 		: ant-colony optimisation implementation

to run each program, use:
"python *filename*", example "python nearest_neighbors.py" (without quotation marks)

each script expects 2 input files in the same directory:
- store information, i.e. "../input/stores.xlsx"
- result from the previous script, e.g. "tabu_search.py" requires "nearest_neighbors.xls"
(therefore, the scripts need to be executed sequentially, except "model.py")

to make the script run for shorter period of time, modify the configuration inside the script.
