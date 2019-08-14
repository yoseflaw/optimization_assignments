import numpy as np
import pandas as pd
from haversine import haversine
from copy import copy

import matplotlib.pyplot as plt

class Maps(object):
    """Store all dictionary regarding the stores' locations and name"""

    def __init__(self, store_file_name):
        """Read store information from a file and calculate all distances between stores"""

        store_df = pd.read_excel(store_file_name, index_col=0)

        # create a column with 1.5 if jumbo, 1 otherwise as visit time
        store_df['visit_time'] = store_df['Type'].apply(lambda t: 1.5 if t == 'Jumbo' else 1)

        distances = {}

        # calculate distances between stores with haversine
        for i in store_df.index.values:
            distances[i] = {}
            store1 = (store_df.loc[i]['Lat'], store_df.loc[i]['Long'])
            for j in store_df.index.values:
                store2 = (store_df.loc[j]['Lat'], store_df.loc[j]['Long'])
                distances[i][j] = haversine(store1, store2)

        self.dist_df = pd.DataFrame(distances) # distance matrix
        self.dist_df.index.name = 'store_id'
        self.visit_time = store_df['visit_time']
        self.store_names = store_df['Name']
        self.store_lat = store_df['Lat']
        self.store_long = store_df['Long']
    
    def get_length(self):
        return len(self.dist_df)
    
    def get_store_name(self, store_id):
        return self.store_names[store_id]
    
    def get_store_list(self):
        return list(self.dist_df.index.unique()[1:])
    
    def get_distance(self, store1, store2):
        return self.dist_df.loc[store1, store2]
    
    def get_visit_time(self, store):
        return self.visit_time[store] if store != 0 else 0
    
    def get_next_nearest_store(self, curr_store_id, visited_stores):
        """function to get the store id of the closest store to current store,
        excluding a set of store ids in visited_stores"""
        return np.argmin(
            self.dist_df.loc[curr_store_id].drop(labels=visited_stores))

class Route(object):
    """Representation of a route in a day.
    A route consists of store ids in the order of visit."""

    def __init__(self):
        self.max_hour = 10              # maximum working hour
        self.avg_speed = 80
        self.curr_hour = 9              # store working hour start from 9AM
        self.store_closing_time = 17    # store working hour end at 5PM

        # initialisation
        self.path = [0]                 # the list of stores visited in this route,
                                        # every day begins from HQ (store id = 0)
        self.total_hour = 0
        self.total_distance = 0
        self.splits = []                # all the possible subtours, 
                                        # prepared for two-edge exchange.

    def __str__(self):
        return "Route={}, Distance={:.2f}, Working Hour={:.2f}, End Hour={:.2f}".format(
            str(self.path),
            self.total_distance,
            self.total_hour,
            self.curr_hour
        )

    def get_curr_store(self):
        return self.path[-1] if len(self.path) > 0 else None

    def set_splits(self):
        """All possibility of splitting the route into two"""
        for i in range(1, len(self.path)):
            self.splits.append(self.split(i))

    def split(self, i):
        return self.path[0:i], self.path[i:len(self.path)]

    def is_equal(self, path):
        # reversed path is also equal
        return self.path == path or self.path == path[::-1]

    def travel_to(self, store_id, dist_to, hour_to, time_at):
        self.total_hour += hour_to + time_at
        self.total_distance += dist_to
        self.path.append(store_id)

    def add_next_store(self, store_id, maps):
        """add new store id to the end of the path.
        Check if it is possible to add the store:
            - working hour constraint
            - store open time constraint
        """
        dist_to = maps.get_distance(self.get_curr_store(), store_id)
        time_at = maps.get_visit_time(store_id)

        # time required to go to the next store
        hour_to = dist_to / self.avg_speed

        # time required to go back to the HQ if the store is visited
        hour_back = maps.get_distance(store_id, 0) / self.avg_speed
        
        # check total time must be less then max working hour per day
        if self.total_hour + hour_to + time_at + hour_back <= self.max_hour:
            # if it is the first store to visit, 
            # then exclude the time required to go to the store.
            # John can depart earlier than 9AM.
            if len(self.path) == 1 and (self.curr_hour + time_at <= self.store_closing_time):
                self.travel_to(store_id, dist_to, hour_to, time_at)
                self.curr_hour += time_at
                return True
            # when it is not the first store
            elif (self.curr_hour + hour_to + time_at <= self.store_closing_time):
                self.travel_to(store_id, dist_to, hour_to, time_at)
                self.curr_hour += hour_to + time_at
                return True
        return False

    def end_day(self, dist_to_hq):
        """At the end of the day, John need to go back to HQ"""
        hour_back = dist_to_hq / self.avg_speed
        self.total_distance += dist_to_hq
        self.total_hour += hour_back
        self.curr_hour += hour_back
        self.path.append(0)

        # because the route build-up has finished,
        # initialise all the possible splits.
        self.set_splits()

    def traverse(self, path, maps):
        """Go through a path and validate if it satisfy the constraints"""

        # if it is neither start nor end at HQ, then it is an invalid path
        if path[0] != 0 or path[-1] != 0:
            return False
        i = 1
        have_time = True

        # repeat until path is finished or no more time left for the next store
        while i < len(path)-1 and have_time:
            next_store_id = path[i]
            have_time = self.add_next_store(next_store_id, maps)
            i += 1

        # if the whole path satisfy all the constraints
        if have_time:
            dist_to_hq = maps.get_distance(self.get_curr_store(), 0)
            self.end_day(dist_to_hq)
        return have_time

class Schedule(object):
    """Representation of the list of routes."""

    def __init__(self):
        self.routes = {}            # list of routes
        self.total_distance = 0
        self.curr_route_id = 0      # to make route IDs

    def __len__(self):
        return len(self.routes)

    def copy(self, schedule):
        self.routes = copy(schedule.routes)
        self.total_distance = schedule.total_distance
        self.curr_route_id = schedule.curr_route_id

    def get(self, route_id):
        return self.routes[route_id]

    def get_multiple(self, route_ids):
        return [self.get(route_id) for route_id in route_ids]

    def pop(self, route_id):
        self.total_distance -= self.routes[route_id].total_distance
        return self.routes.pop(route_id)

    def pop_multiple(self, route_ids):
        return [self.pop(route_id) for route_id in route_ids]

    def add(self, route):
        """Adding new route means adding more distance"""
        new_route_id = None
        # add only if there is at least 1 store to be visited,
        # otherwise ignore
        if len(route.path) > 2:
            self.routes[self.curr_route_id] = route
            new_route_id = self.curr_route_id
            self.curr_route_id += 1
            self.total_distance += route.total_distance
        return new_route_id

    def replace(self, route_id, new_route):
        """Replace existing route with new route"""
        old_route = self.pop(route_id)
        if len(new_route.path) > 2:
            self.routes[route_id] = new_route
            self.total_distance += new_route.total_distance
            return True
        return False

    def add_multiple(self, routes):
        return [self.add(route) for route in routes]

    def get_all_route_ids(self):
        return list(self.routes.keys())

    def draw(self, maps, random_seed=1337):
        """Helper function to draw the routes on a 2d plot"""
        plt.figure(figsize=(15, 12))
        for k in self.routes:
            random_color = np.random.rand(3)
            route = self.routes[k]
            plt.plot(
                maps.store_long[route.path], maps.store_lat[route.path], color=random_color, alpha=0.3)
            plt.scatter(maps.store_long[route.path],
                        maps.store_lat[route.path], color=random_color)
        plt.show()

    def export_to_xls(self, file_name, maps):
        results = []
        total_distance = 0
        route_nr = 1
        for route_id in self.routes:
            path = self.routes[route_id].path
            distance = 0
            results.append(
                (route_nr, path[0], maps.get_store_name(
                    path[0]), 0.0, round(total_distance,2))
            )
            for i in range(1, len(path)):
                distance += maps.get_distance(path[i-1], path[i])
                total_distance += maps.get_distance(path[i-1], path[i])
                results.append(
                    (route_nr, path[i], maps.get_store_name(
                        path[i]), round(distance,2), round(total_distance,2))
                )
            route_nr += 1
        df = pd.DataFrame(results, columns=[
            'Route Nr.', 'City Nr.', 'City Name',
            'Total Distance in Route (km)', 'Total Distance (km)'])
        df.to_excel(file_name, index=None, header=True)
        return df

    def read_from_xls(self, file_name, maps):
        all_store_ids = list(maps.dist_df.index.values)
        schedule_df = pd.read_excel(file_name)
        path_df = schedule_df.groupby('Route Nr.')['City Nr.'].apply(
            lambda x: ','.join(x.astype(str))).reset_index()

        for _, row in path_df.iterrows():
            path = []
            for store in row['City Nr.'].split(','):
                store_id = int(store)
                path.append(store_id)
                if store_id != 0:
                    if store_id in all_store_ids:
                        all_store_ids.remove(store_id)
                    else:
                        print("[ERROR] Store-{} visited more than once".format(store_id))
                        return False
            route = Route()
            is_valid = route.traverse(path, maps)
            if not is_valid:
                print("[ERROR] Route {} is invalid".format(path['Route Nr.']))
                return False
            else:
                self.add(route)
        if len(all_store_ids) > 1:
            print("[ERROR] Store IDs not visited: {}".format(all_store_ids[1:]))
            return False
        return True