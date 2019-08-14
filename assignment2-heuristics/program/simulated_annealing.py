import imp
from simanneal import Annealer
import pants
import random

from model import Maps, Route, Schedule
import two_edge_exchange as tex

class VRPAnnealer(Annealer):
    """Implementation of simulated annealing, class inherits from simanneal.
    """

    def __init__(self, schedule, maps, config=None):

        # initialise parent class Annealer with TwoEdgeExchange object as state
        super(VRPAnnealer, self).__init__(tex.TwoEdgeExchange(schedule, maps))

        # set simanneal configuration, if config is None then use default
        if config is not None:
            self.steps = config.get('steps', 10000)
            self.Tmax = config.get('Tmax', 10)
            self.Tmin = config.get('Tmin', 1)
            self.updates = config.get('updates', 100)
        self.copy_strategy = "deepcopy"

    # override next neighbor for random routes
    def next_neighbor(self):
        """get the next neighbor.
        For simulated annealing, the neighborhood structure can be either
        2-edge exchange or city swap between 2 random routes. Except if the
        random routes turn out to be the same, then use single route exchange.

        The city swap is implemented as an additional possibility of getting
        a neighbor because doing only 2-edge exchange did not produce better
        solution after being optimised by the tabu search.
        """
        all_route_ids = self.state.schedule.get_all_route_ids()
        delta, neighbor = None, None
        exchange_id = (None, None)
        while delta is None:
            route_id1 = all_route_ids[random.randint(0, len(all_route_ids)-1)]
            route_id2 = all_route_ids[random.randint(0, len(all_route_ids)-1)]
            if route_id1 == route_id2:
                delta, neighbor = self.state.single_route_exchange(route_id1)
                exchange_id = (route_id1,)
                
            # a way to randomise between 2-edge exchange and city swap
            # because route_id1 and route_id2 are random
            elif route_id1 < route_id2:
                delta, neighbor = self.state.two_route_exchange(route_id1, route_id2)
                exchange_id = (route_id1, route_id2)
            else:
                delta, neighbor = self.state.city_swap(route_id1, route_id2)
                exchange_id = (route_id1, route_id2)

        return delta, exchange_id, neighbor

    def update_schedule(self, delta, exchange_id, new_neighbor):
        if len(exchange_id) != len(new_neighbor):
            print("[ERROR] TwoEdgeExchange: inequal number of new routes: {} {}".format(exchange_id, new_neighbor))
            return False

        return [self.state.schedule.replace(exchange_id[i], new_neighbor[i]) for i in range(len(exchange_id))]

    def move(self): 
        # pick random 2-edge exchange as neighbor
        delta, exchange_id, neighbor = self.next_neighbor()

        # execute 2-edge exchange
        self.update_schedule(delta, exchange_id, neighbor)

    def energy(self):
        return self.state.schedule.total_distance


def main():
    maps_file = '../input/stores.xlsx'
    initial_solution_file = '../results/3-tabu_search.xls'
    output_file = '../results/4-simulated_annealing.xls'

    print("[INFO] Reading store information from: {}".format(maps_file))
    maps = Maps(maps_file)

    print("[INFO] Reading initial solution from: {}".format(initial_solution_file))
    schedule = Schedule()
    valid_solution = schedule.read_from_xls(initial_solution_file, maps)
    if not valid_solution:
        print("[ERROR] Abort: initial solution contains invalid route(s)")
        return False
    else:
        random.seed(1337)
        print("[INFO] Start Simulated Annealing")
        ## I use this configuration to get the result,
        ## but it will take 1 hour to run
        # config = {
        #     "steps": 50000,
        #     "Tmax": 10,
        #     "Tmin": 1,
        #     "updates": 100
        # }

        # configuration used to test
        config = {
            "steps": 1000,
            "Tmax": 5,
            "Tmin": 1,
            "updates": 10
        }

        sima = VRPAnnealer(schedule, maps, config)
        tex, e = sima.anneal()
        print("[INFO] Simulated Annealing best schedule score: {}".format(e))

        print("[INFO] Export solution to: {}".format(output_file))
        tex.schedule.export_to_xls(output_file, maps)

if __name__ == "__main__":
    main()