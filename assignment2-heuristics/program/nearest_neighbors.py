import numpy as np
import pandas as pd
import math
from model import Maps, Route, Schedule

class NearestNeighbors(object):
    """Implementation of nearest neighbors algorithm.
    At every iteration, consider the closest store (by distance only) from 
    current store position. Add only if it still satisfies all route constraints.
    """
    def __init__(self, maps):
        self.maps = maps

    def solve(self):
        # get the store list, make sure all stores are visited
        stores_list = self.maps.get_store_list()
        visited_stores = [0]
        schedule = Schedule()

        # while there is still store to visit
        # outer loop is to iterate over the routes
        while (len(stores_list) > 0):
            route = Route()
            have_time = True

            # inner loop is to build a route
            while have_time and len(stores_list) > 0:
                curr_store_id = route.get_curr_store()
                next_store_id = self.maps.get_next_nearest_store(
                    curr_store_id, visited_stores)

                # check if the closest store is feasible
                # to be visited
                have_time = route.add_next_store(
                    next_store_id,
                    self.maps
                )

                # if it is visited
                if have_time:
                    visited_stores.append(next_store_id)
                    stores_list.remove(next_store_id)

            # finish route by going back to HQ
            dist_to_hq = self.maps.get_distance(route.get_curr_store(), 0)
            route.end_day(dist_to_hq)

            schedule.add(route)
        return schedule

def main():
    maps_file = '../input/stores.xlsx'
    output_file = '../results/1-nearest_neighbors.xls'
    print("[INFO] Reading store information from: {}".format(maps_file))
    maps = Maps(maps_file)
    print("[INFO] Start solving schedule with nearest neighbors algorithm")
    nn = NearestNeighbors(maps)
    schedule = nn.solve()
    print("[INFO] Nearest Neighbors score: {:.2f}".format(schedule.total_distance))
    print("[INFO] Export solution to: {}".format(output_file))
    schedule.export_to_xls(output_file, maps)

if __name__ == "__main__":
    main()