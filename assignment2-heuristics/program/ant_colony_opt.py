import imp
from simanneal import Annealer
import pants
import random

from model import Maps, Route, Schedule

class ACO(object):
    """Implementation of ant-colony optimisation (ACO).
    ACO optimises a route instead of the whole schedule.
    """

    def __init__(self, maps):
        self.maps = maps
        
    def make_cycle(self, path):
        """rotate path, HQ should be the last position.
        Re-append HQ also as starting point.
        """

        # put HQ as the start position
        while path[-1] != 0:
            path = path[1:] + path[:1]
        # put HQ as the end position
        return [0] + path
    
    def optimize(self, route):
        # exclude one HQ (otherwise there will be 2 HQs in the path)
        nodes = route.path[:len(route.path)-1]

        world = pants.World(nodes, self.maps.get_distance)
        solver = pants.Solver()
        return solver.solve(world)
    
    def optimize_all(self, schedule):
        # call optimize() for each route
        for uid in schedule.get_all_route_ids():
            curr_route = schedule.get(uid)
            solution = self.optimize(curr_route)

            # rotate the route
            new_path = self.make_cycle(solution.tour)
            new_route = Route()

            # make sure the new route is feasible
            if new_route.traverse(new_path, self.maps):
                schedule.replace(uid, new_route)

            # if it is not feasible, use unoptimised route
            else:
                print("[INFO] route is no longer valid after ACO: {}".format(new_route))

def main():
    maps_file = '../input/stores.xlsx'
    initial_solution_file = '../results/4-simulated_annealing.xls'
    output_file = '../results/5-ant_coloy_opt.xls'

    print("[INFO] Reading store information from: {}".format(maps_file))
    maps = Maps(maps_file)

    print("[INFO] Reading initial solution from: {}".format(initial_solution_file))
    schedule = Schedule()
    valid_solution = schedule.read_from_xls(initial_solution_file, maps)
    if not valid_solution:
        print("[ERROR] Abort: initial solution contains invalid route(s)")
        return False
    else:
        print("[INFO] Start ACO to optimize each route")
        aco = ACO(maps)
        aco.optimize_all(schedule)
        print("[INFO] ACO best schedule score: {}".format(schedule.total_distance))

        print("[INFO] Export solution to: {}".format(output_file))
        ## Printing and drawing solution
        # for k in schedule.routes:
        #     print(schedule.routes[k])
        # schedule.draw(maps)
        schedule.export_to_xls(output_file, maps)

if __name__ == "__main__":
    main()