import imp
from copy import deepcopy
from model import Maps, Route, Schedule

import matplotlib.pyplot as plt
import two_edge_exchange as tex

class TabuSearch(tex.TwoEdgeExchange):
    """Implementation of tabu search agorithm, 
    use TwoEdgeExchange as a base class because TabuSearch use
    2-edge exchange as neighborhood structure.
    
    Tabu search specification:
    - Solution          : a Schedule object
    - Initial solution  : nearest neighbors, optimized with TwoEdgeExchange
    - Neighborhood      : all possible 2-edge exchange of every edge 
                        combination of the current schedule, 
                        including both single and two route exchange.
    - Tabu List:        : a dictionary where exchange between route i and route j 
                        that happened recently (determined by tabu_period) is 
                        stored in the tabu_dict[i, j] if i < j, otherwise
                        stored in the tabu_dict[j, i] (both i and j are every 
                        element of routes in schedule).
    - Aspiration        : Allow a move, even if it is tabu, if the total distance
                        is less then the distance of the current best-known solution.
    - Termination       : No improvement for a pre-defined number of iterations or
                        number of iterations has exceeded a maximum threshold.

    """
    def __init__(self, schedule, maps, tabu_period=10):
        # initialise parent class (TwoEdgeExchange)
        super(TabuSearch, self).__init__(schedule, maps)

        self.tabu_period = tabu_period

        # initialisation for tabu list and best-known solution
        self.tabu_dict = {}
        self.best_schedule = deepcopy(schedule)
    
    def add_tabu(self, route_ids):
        """add recently exchanged route IDs to tabu list
        depend on the number of route ID(s) and which
        route ID is bigger.
        """
        if len(route_ids) == 1:
            route_ids = (route_ids[0], route_ids[0])
        elif route_ids[0] > route_ids[1]:
            route_ids = (route_ids[1], route_ids[0])
        self.tabu_dict[route_ids] = self.tabu_period
        
    def update_tabu(self):
        """update tabu list:
        if tabu value equal 1, remove from tabu list.
        Else, decrease tabu value by 1.
        """
        times_up = []
        
        for ids in self.tabu_dict:
            if self.tabu_dict[ids] == 1:
                times_up.append(ids)
            elif self.tabu_dict[ids] > 1:
                self.tabu_dict[ids] -= 1
        
        for ids in times_up:
            self.tabu_dict.pop(ids)

    # Override next neighbor method to include tabu consideration
    def next_neighbor(self):
        """get the next neighbor.
        For tabu search, pick the neighbor with the highest gain.
        If there is no neighbor that gives improvement, pick the lowest lost.
        Return both scenario: 
            - max_result[0] = all possible neighbors for aspiration criteria
            - max_result[1] = only non-tabu neighbors
        """

        if self.neighbors is None:
            return None

        # initialisation
        max_delta, max_delta_tabu = None, None
        min_neighbor, min_neighbor_tabu = None, None
        exchange_id, exchange_id_tabu = None, None

        for route_id1 in self.neighbors:
            for route_id2 in self.neighbors[route_id1]:
                delta, neighbor = self.neighbors[route_id1][route_id2]

                # update best neighbor depending on the delta
                if max_delta is None or (delta is not None and delta > max_delta):
                    max_delta = delta
                    exchange_id = (route_id1,) if route_id1 == route_id2 else (
                        route_id1, route_id2)
                    min_neighbor = neighbor

                # update best non-tabu neighbor
                # only consider routes combination which does not exist in tabu list
                if (route_id1, route_id2) not in self.tabu_dict.keys() and (
                        max_delta_tabu is None or (
                        delta is not None and delta > max_delta_tabu)):
                    max_delta_tabu = delta
                    exchange_id_tabu = (route_id1,) if route_id1 == route_id2 else (
                        route_id1, route_id2)
                    min_neighbor_tabu = neighbor

        max_result = [
            (max_delta, exchange_id, min_neighbor),                 # best move of all
            (max_delta_tabu, exchange_id_tabu, min_neighbor_tabu)   # best move of non-tabu
        ]
        return max_result
                
    # Override two edge exchange solve() method
    def solve(self, max_iter=10, max_no_improvement=5):
        """start tabu search, stopping criteria can be:
            - max_iter: maximum number of iteration
            - max_no_improvement: maximum number of iteration with no improvement

        If either stopping criteria is not used, set to None.
        Both cannot be None simultaneously.
        """

        if max_iter is None and max_no_improvement is None:
            print("[ERROR] Either max_iter or max_no_improvement has to be set as stopping criterium")
            return False

        if max_iter is not None and max_iter < 1:
            print("[ERROR] max_iter cannot be smaller than 1")
            return False

        if max_no_improvement is not None and max_no_improvement < 1:
            print("[ERROR] max_no_improvement cannot be smaller than 1")
            return False

        log = []
        i = 0
        count_no_improvement = 0

        # calculate the neighbors for first iteration
        # store every possible 2-edge exchange between all edges
        # as the neighborhood
        self.initialise_neighbors()
        while (max_iter is None or i < max_iter) and (max_no_improvement is None or count_no_improvement < max_no_improvement):
            max_result = self.next_neighbor()
            # check aspiration criteria with the best out of all neighbors
            delta, exchange_id, neighbor = max_result[0]
             
            next_distance = self.schedule.total_distance - delta
            if next_distance < self.best_schedule.total_distance:
                new_ids = self.update_schedule(delta, exchange_id, neighbor)
                self.best_schedule = deepcopy(self.schedule)
            
            # not aspiration criteria, get the best non-tabu move
            else:
                delta, exchange_id, neighbor = max_result[1]
                self.update_schedule(delta, exchange_id, neighbor)

            self.update_tabu()
            self.add_tabu(exchange_id)
            log.append(self.schedule.total_distance)

            # reset no improvement count if delta is positive (distance reduced)
            # use 0.01 to prevent float problem
            count_no_improvement = 0 if max_no_improvement is None or delta > 0.01 else (
                count_no_improvement + 1)
            i += 1

            if (max_iter is not None and max_iter > 20 and i % (max_iter // 10) == 0) or (
                max_iter is None and max_no_improvement is not None and i % max_no_improvement == 0):
                print('.', end='', flush=True)
            if (max_iter is not None and max_iter > 20 and i == max_iter) or (
                max_no_improvement is not None and count_no_improvement == max_no_improvement):
                print()
        return True

def main():
    maps_file = '../input/stores.xlsx'
    initial_solution_file = '../results/2-two_edge_exchange.xls'
    output_file = '../results/3-tabu_search.xls'

    print("[INFO] Reading store information from: {}".format(maps_file))
    maps = Maps(maps_file)

    print("[INFO] Reading initial solution from: {}".format(initial_solution_file))
    schedule = Schedule()
    valid_solution = schedule.read_from_xls(initial_solution_file, maps)

    if not valid_solution:
        print("[ERROR] Abort: initial solution contains invalid route(s)")
        return False
    else:
        print("[INFO] Start improving schedule with Tabu Search")
        ts = TabuSearch(schedule, maps, tabu_period=50)

        ## I use this configuration to get the result,
        ## but it will take approx. 1 hour to run
        # if not ts.solve(max_iter=5000, max_no_improvement=100):
        #     return False

        # configuration used for testing
        if not ts.solve(max_iter=100, max_no_improvement=None):
            return False
            
        schedule = ts.best_schedule
        print("[INFO] Tabu search best score: {:.2f}".format(schedule.total_distance))
        
        ## Plotting score for each iterations
        # plt.plot(ts.log)
        # plt.show()
        print("[INFO] Export solution to: {}".format(output_file))
        schedule.export_to_xls(output_file, maps)
        return True

if __name__ == "__main__":
    main()