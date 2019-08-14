import random
from pprint import pprint

from model import Maps, Route, Schedule

class TwoEdgeExchange(object):
    """Implementation of 2-edge exchange algorithm.
    There are two types of 2-edge exchange:
        - single route exchange: 2 edges of the same route
        - two route exchange: 2 edges of 2 different routes

    At every iteration: 
        1. consider all possible 2-edge exchange (neighbors)
            within route and between routes.
        2. pick a neighbor which gives the highest gain
        3. update the neighbors according to the chosen neighbor

    Stopping criteria: 
        1. no improvement is available among all the neighbors.
        2. number of iteration reach the maximum allowed (max_iter)
    """

    def __init__(self, schedule, maps):
        self.schedule = schedule
        self.maps = maps
        self.neighbors = None

    def single_route_exchange(self, route_id, pick_random=False):
        """2-edge exchange within a route"""

        route = self.schedule.get(route_id)

        # no possible exchange can be made 
        # if none or only 1 store is visited
        if len(route.path) < 4:
            return (None, None)

        old_distance = route.total_distance
        neighbor = None
        feasible_neighbors = []
        delta_distance = None

        # go through the stores in the path
        for i in range(1, len(route.path)):

            # edges cannot be adjacent to each other
            for j in range(i+2, len(route.path)):

                # reverse the path in the middle
                candidate = route.path[0:i] + route.path[i:j][::-1] + route.path[j:len(route.path)]

                # check if the route is equal to the original route,
                # a reversed order path is onsidered equal.
                if not route.is_equal(candidate):
                    new_route = Route()

                    # check if new route is feasible
                    # add 0.01 as minimum distance gained to prevent numerical problem (float problem)
                    if new_route.traverse(candidate, self.maps):
                        if pick_random:
                            feasible_neighbors.append((new_route.total_distance, (new_route,)))
                        elif abs(old_distance - new_route.total_distance) > 0.01 and (
                            neighbor is None or new_route.total_distance < neighbor[0].total_distance):
                            # put inside iterable for consistency
                            neighbor = (new_route,)

        if pick_random and len(feasible_neighbors) > 0:
            delta_distance, neighbor = feasible_neighbors[random.randint(0,len(feasible_neighbors)-1)]
        elif not pick_random and neighbor is not None:
            delta_distance = old_distance - neighbor[0].total_distance

        return (delta_distance, neighbor)

    def two_route_exchange(self, route_id1, route_id2, pick_random=False):
        """2-edge exchange between 2 routes"""

        if route_id1 == route_id2:
            print("[ERROR] Route IDs must be different for two_route_exchange: {} {}".format(
                route_id1, route_id2))
        min_dist = None
        neighbor = None
        feasible_neighbors = []
        delta_distance = None
        route1, route2 = self.schedule.get_multiple((route_id1, route_id2))
        old_distance = route1.total_distance + route2.total_distance

        # go through every possible split of each route
        for split1 in route1.splits:
            for split2 in route2.splits:

                # for every 2-edge exchange, there are 2 alternatives
                # each alternative consists of possibly 2 routes.
                # e.g. we want to exchange:
                #   route1 = [ABC][DEF], route2 = [GHI][JKL]
                candidates = (
                    # alternative 1: [ABC][JKL], [GHI][DEF]
                    (split1[0] + split2[1], split2[0] + split1[1]),

                    # alternative 2: [ABC][IHG], [LKJ][DEF]
                    (split1[0] + split2[0][::-1], split1[1][::-1] + split2[1])
                )

                # check feasibility and find minimum
                for candidate in candidates:
                    new_route1 = Route()

                    # check if the first candidate route is equal with either route 1 or 2
                    is_exist1 = route1.is_equal(
                        candidate[0]) or route2.is_equal(candidate[0])

                    # check if the first candidate route is feasible
                    if not is_exist1 and new_route1.traverse(candidate[0], self.maps):
                        new_route2 = Route()
                        
                        # check if the second candidate route is equal with either route 1 or 2
                        is_exist2 = route1.is_equal(
                            candidate[1]) or route2.is_equal(candidate[1])

                        # check if the second route is feasible
                        if not is_exist2 and new_route2.traverse(candidate[1], self.maps):
                            new_distance = new_route1.total_distance + new_route2.total_distance
                            if pick_random:
                                feasible_neighbors.append((new_distance - old_distance,
                                    (new_route1, new_route2)
                                ))

                            # if pick_random is False,
                            # consider the best out of the feasible alternatives
                            # in terms of distance.
                            elif (min_dist is None or (
                                new_route1.total_distance + new_route2.total_distance < min_dist)):
                                min_dist = new_distance
                                neighbor = (new_route1, new_route2)

                        # if is_valid2 and (min_dist is None or (
                        #         new_route1.total_distance + new_route2.total_distance < min_dist)):
                        #     min_dist = new_route1.total_distance + new_route2.total_distance
                        #     min_neighbor = (new_route1, new_route2)
        
        if pick_random and len(feasible_neighbors) > 0:
            delta_distance, neighbor = feasible_neighbors[random.randint(0,len(feasible_neighbors)-1)]
        elif not pick_random and min_dist is not None:
            delta_distance = old_distance - min_dist
        
        return (delta_distance, neighbor)


    def city_swap(self, route_id1, route_id2, pick_random=False):
        """city swap in two route"""
        if route_id1 == route_id2:
            print("[ERROR] Route IDs must be different for city_swap: {} {}".format(
                route_id1, route_id2))

        route1, route2 = self.schedule.get_multiple([route_id1, route_id2])

        old_distance = route1.total_distance + route2.total_distance
        min_dist = None
        neighbor = None
        feasible_neighbors = []
        delta_distance = None

        # go through each store in route 1 except HQ
        for idx1, store1 in enumerate(route1.path[1:len(route1.path)-1]):

            new_path1 = route1.path[:]

            # go through each store in route 2 except HQ
            for idx2, store2 in enumerate(route2.path[1:len(route2.path)-1]):

                new_path2 = route2.path[:]

                # swap city between two routes
                new_path1[idx1+1], new_path2[idx2+1] = store2, store1

                # check if new routes are feasible
                # add 0.01 as minimum distance gained to prevent numerical problem (float problem)
                new_route1 = Route()
                if new_route1.traverse(new_path1, self.maps):
                    new_route2 = Route()
                    if new_route2.traverse(new_path2, self.maps):
                        new_distance = new_route1.total_distance + new_route2.total_distance
                        if pick_random:
                            feasible_neighbors.append((old_distance - new_distance, 
                                (new_route1, new_route2))
                            )
                        # if pick_random is False,
                        # consider the best out of the feasible alternatives in terms of distance.
                        elif (min_dist is None or (
                            new_route1.total_distance + new_route2.total_distance < min_dist)):
                            min_dist = new_distance
                            neighbor = (new_route1, new_route2)    
        
        if pick_random and len(feasible_neighbors) > 0:
            delta_distance, neighbor = feasible_neighbors[random.randint(0,len(feasible_neighbors)-1)]
        elif not pick_random and min_dist is not None:
            delta_distance = old_distance - min_dist
        
        return (delta_distance, neighbor)     


    def initialise_neighbors(self):
        """Before the first iterations,
        calculate all possible 2-edge exchange of the initial solution"""
        self.neighbors = {}
        route1_ids = self.schedule.get_all_route_ids()

        for route1_id in route1_ids:
            # 2-edge exchange within route
            if route1_id not in self.neighbors:
                single_exchange = self.single_route_exchange(route1_id)
                self.neighbors[route1_id] = {route1_id: single_exchange}

            # 2-edge exchange with all other routes
            route2_ids = [
                route2_id for route2_id in route1_ids if route2_id > route1_id]
            for route2_id in route2_ids:
                delta, neighbor = self.two_route_exchange(route1_id, route2_id)
                if delta is not None:
                    self.neighbors[route1_id][route2_id] = (delta, neighbor)

    def next_neighbor(self):
        """get the next neighbor among all possible neighbors.
        For 2-edge exchange, pick the neighbor with the highest distance reduction
        """
        if self.neighbors is None:
            return None

        max_delta = None
        min_neighbor = None
        exchange_id = None
        for route_id1 in self.neighbors:
            for route_id2 in self.neighbors[route_id1]:
                delta, neighbor = self.neighbors[route_id1][route_id2]
                if max_delta is None or (delta is not None and delta > max_delta):
                    max_delta = delta
                    exchange_id = (route_id1,) if route_id1 == route_id2 else (
                        route_id1, route_id2)
                    min_neighbor = neighbor

        return max_delta, exchange_id, min_neighbor


    def update_schedule(self, delta, exchange_id, new_neighbor):
        """Given the next chosen neighbor, 
        update the schedule and the 2-edge exchange neighbors list"""

        # make sure that the number of new neighbors candidate is the same
        # with the number of routes exchanged
        if len(exchange_id) != len(new_neighbor):
            print("[ERROR] TwoEdgeExchange: inequal number of new routes: {} {}".format(
                exchange_id, new_neighbor))
            return False

        exchange_valid_ids = []
        removed_ids = []
        for i in range(len(exchange_id)):
            route_id = exchange_id[i]
            is_valid = self.schedule.replace(route_id, new_neighbor[i])

            # make sure the route is not removed (because possible route=[0,0])
            if is_valid:
                exchange_valid_ids.append(route_id)

                # conduct single route exchange to every new route
                delta, neighbor = self.single_route_exchange(route_id)
                if delta is not None:
                    self.neighbors[route_id] = {route_id: (delta, neighbor)}
                else:
                    self.neighbors[route_id] = {}
            else:
                self.neighbors.pop(route_id)
                removed_ids.append(route_id)

        route_ids = self.schedule.get_all_route_ids()

        # for every existing route, calculate 2-edge exchange with the new routes
        for route_id in route_ids:
            for removed_id in removed_ids:
                if route_id < removed_id:
                    self.neighbors[route_id].pop(removed_id)

            for new_id in exchange_valid_ids:
                if route_id != new_id:
                    uid1, uid2 = (route_id, new_id) if route_id < new_id else (new_id, route_id)
                    delta, neighbor = self.two_route_exchange(uid1, uid2)
                    if delta is not None:
                        self.neighbors[uid1][uid2] = (delta, neighbor)
                    elif uid2 in self.neighbors[uid1]:
                        self.neighbors[uid1].pop(uid2)

        return exchange_id

    def solve(self, max_iter=100):
        # calculate all possible 2-edge exchanges in the schedule
        self.initialise_neighbors()

        # find the next neighbor.
        # In this case, the one with the highest distance reduction
        delta, exchange_id, neighbor = self.next_neighbor()
        log = []
        i = 0

        # until there is no improvement or reach maximum number of iteration
        while delta > 0 and i < max_iter:
            # execute the 2-edge exchange and update the neighbors
            self.update_schedule(delta, exchange_id, neighbor)
            log.append(self.schedule.total_distance)

            # continue and find the next neighbor
            delta, exchange_id, neighbor = self.next_neighbor()
            i += 1
        return log

def main():
    maps_file = '../input/stores.xlsx'
    initial_solution_file = '../results/1-nearest_neighbors.xls'
    output_file = '../results/2-two_edge_exchange.xls'

    print("[INFO] Reading store information from: {}".format(maps_file))
    maps = Maps(maps_file)

    print("[INFO] Reading initial solution from: {}".format(initial_solution_file))
    schedule = Schedule()
    valid_solution = schedule.read_from_xls(initial_solution_file, maps)

    if not valid_solution:
        print("[ERROR] Abort: initial solution contains invalid route(s)")
        return False
    else:
        print("[INFO] Initialise neighbors for 2-edge exchange")
        tex = TwoEdgeExchange(schedule, maps)
        print("[INFO] Start improving with 2-edge exchange")
        log = tex.solve()
        print("[INFO] 2-edge exchange best score: {:.2f}".format(schedule.total_distance))
        print("[INFO] Export solution to: {}".format(output_file))
        schedule.export_to_xls(output_file, maps)

if __name__ == "__main__":
    main()