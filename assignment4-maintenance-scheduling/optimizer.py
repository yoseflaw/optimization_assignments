import numpy as np
import pandas as pd
from pulp import *

#global class that solves the engine maintenance optimization task
class Optimizer(object):

    #optimizer constructor, the 2 optional parameters are used for optimization task 2 and 3 respectively
    def __init__(self, time_horizon, rul_filename, work_time_filename, teams_filename, \
                    cost_filename, max_engine_filename=None, waiting_time_filename=None):
        self.T = time_horizon
        self.ruls = self.build_rul(rul_filename, self.T)
        self.engine_ids = list(self.ruls.keys())

        self.work_times = self.build_work_time(work_time_filename, teams_filename)

        self.team_ids = []
        for team_id, _ in self.work_times.keys():
            if team_id not in self.team_ids:
                self.team_ids.append(team_id)

        self.team_types = self.build_team_types(teams_filename)

        self.costs = self.build_costs(cost_filename)

        self.max_engine = self.build_max_engine(max_engine_filename, 
            teams_filename) if max_engine_filename is not None else None
        self.waiting_time = self.build_waiting_time(waiting_time_filename, 
            teams_filename) if waiting_time_filename is not None else None

        self.engine_costs = self.calculate_engine_costs()
        self.potential_saved_costs = self.calculate_saved_costs()

        self.model = None

    #Purpose: given a time horizon filters and keeps only the engines to be examined 
	#Returns: dictionary with engines and their RUL
    def build_rul(self, rul_filename, time_horizon):
        base_rul = pd.read_excel(rul_filename, headers=0)
        n_rul = base_rul.where(base_rul['RUL'] <= time_horizon)
        rul_dict = {}
      
        for _, row in n_rul.iterrows():
            engine_id = row['id']
            if not np.isnan(engine_id):
                rul_dict[int(engine_id)] = int(row['RUL'])
          
        return rul_dict

    #Purpose: given a file path builds a dictionary with the corresponding teams and their types 
    def build_team_types(self, teams_filename):
        base_teams = pd.read_csv(teams_filename)
        team_type_dict = {}
        for _, row in base_teams.iterrows():
            team_type_dict[row['team_id']] = row['type']
        return team_type_dict

    #Purpose: builds the dictionary containing information about:
	#How many maintenance days are needed by each team for each engine. 
    def build_work_time(self, work_time_filename, teams_filename):
        base_work_time = pd.read_csv(work_time_filename) 
        base_teams = pd.read_csv(teams_filename)

        temp_work_time = base_work_time[base_work_time['engine_id'].isin(self.engine_ids)]
        teams_work_time = base_teams.merge(temp_work_time, on='type', how='left')

        work_time_dict = {}
        for _, row in teams_work_time.iterrows():
            work_time_dict[row['team_id'], row['engine_id']] = row['numberofdays']
        return work_time_dict

    #Purpose: builds the dictionary containing information about:
	#The daily cost for each engine if maintained after RUL expiration 
    def build_costs(self, cost_filename):
        base_costs = pd.read_csv(cost_filename)
        temp_costs = base_costs[base_costs['engine_id'].isin(self.engine_ids)]
      
        costs_dict = {}
        for _, row in temp_costs.iterrows():
            costs_dict[row['engine_id']] = row['cost']
      
        return costs_dict

    #Purpose: includes working time for each team. This is used in optimization task 3
    def build_waiting_time(self, waiting_time_filename, teams_filename):
        base_waiting_time = pd.read_csv(waiting_time_filename)
        base_teams = pd.read_csv(teams_filename)

        teams_waiting_time = base_teams.merge(base_waiting_time, on='type', how='left')

        waiting_time_dict = {}
        for _, row in teams_waiting_time.iterrows():
            waiting_time_dict[row['team_id']] = row['waiting_time']
        return waiting_time_dict

    #Purpose: builds the dictionary containing the max number of engines that a team can work during a planning horizon
    def build_max_engine(self, max_engine_filename, teams_filename):
        base_max_engine = pd.read_csv(max_engine_filename)
        base_teams = pd.read_csv(teams_filename)

        teams_max_engine = base_teams.merge(base_max_engine, on='type', how='left')

        max_engine_dict = {}
        for _, row in teams_max_engine.iterrows():
            max_engine_dict[row['team_id']] = row['max_engine']
        return max_engine_dict

    #Purpose: Builds a dictionary containing information for the maintenance cost applied to each engine at day t 
	#if the RUL has expired at time t for the examined engine
    def calculate_engine_costs(self):
        engine_costs = {}
        for j in self.engine_ids:
            for t in range(1,self.T+1):
                engine_costs[j,t] = self.costs[j] if t > self.ruls[j] else 0
      
        return engine_costs    

    #Purpose: calculates the saved cost for each engine j maintained from team i at day t
	#It is the s variable from the presented formulation in optimization task 1 Part A
    def calculate_saved_costs(self): 
        saved_cost = {}
        #print(self.team_ids)
        #print(self.engine_ids)
        for i in self.team_ids:
            for j in self.engine_ids:
                for t in range(1,self.T):
                    saved_cost[i,j,t] = (self.T - max(t + self.work_times[i,j]-1, self.ruls[j])) * self.costs[j]
                    # saved_cost[(i,new_rul.iloc[j-1]['engine_id'],t)] = (T - max(t + new_work_time.iloc[i-1]['numberofdays']-1, new_rul.iloc[j-1]['RUL'])) * new_costs.iloc[j-1]['cost']
            
        return saved_cost

    #Purpose: Pre-compute the first of maintenance for each examined engine, for all teams for all days
	#This function is used for the second constraint 
    def precompute_first_day_maintenance(self):
        first_day = {}
        for i in self.team_ids:
            for j in self.engine_ids:
                for t in range(1,self.T):
                    w_i = 0 if self.waiting_time is None else self.waiting_time[i]
                    first_day[i,j,t] = max(t - w_i - self.work_times[i,j] + 1,1)
        
        return first_day

    #Purpose: Using pulp builds the variables, activation function and constrains and solves the optimization problem 
    def solve(self):
	
        #variables declaration for optimization problem
        T = self.T
        I = self.team_ids
        J = self.engine_ids
        mu = self.work_times
        theta = list(range(1, self.T))
        s = self.potential_saved_costs
        d = self.precompute_first_day_maintenance()
        k = self.max_engine

        self.model = LpProblem(name='assignment4', sense=LpMaximize)

        # Decision Variable
        self.x = LpVariable.dicts("x", [(i,j,t) 
                                   for i in I
                                   for j in J
                                   for t in theta],0,1,LpBinary)

        # Objective function
        self.model += lpSum(self.x[i,j,t] * s[i,j,t] 
                       for i in I
                       for j in J 
                       for t in theta)

        # Constraint 1, each team at day t can work only on 1 engine
        for i in I:
            for t in theta:
                self.model += lpSum(self.x[i,j,t]
                              for j in J
                              ) <= 1

        # Constrain 2, a team must finish maintenance of an engine before starts maintaining another
        for i in I:
            for t in theta:
                self.model += lpSum(self.x[i,j,t_tone]
                               for j in J
                               for t_tone in range(d[i,j,t], t+1)
                              ) <= 1

        # Constraint 3, every engine must be maintained only by one team
        for j in J:
            self.model += lpSum(self.x[i,j,t]
                          for i in I
                          for t in theta
                          ) <= 1

        # Constraint 4, any selected maintenance should be finished within a time horizon
        self.model += lpSum(self.x[i,j,t] 
                       for i in I
                       for j in J
                       for t in range(T - mu[i, j] + 1, T)
                      ) == 0

        # Constraint 5, each team can work on max number of engines within a time horizon
        if k is not None:
            for i in I:
                self.model += lpSum(self.x[i,j,t]
                              for j in J
                              for t in theta
                              ) <= k[i]

        # Solving
        return LpStatus[self.model.solve()]

    #Purpose: returns a dataframe which contains the the saved cost for all examined engines within planning horizon
    def get_schedule(self):
        schedule = {}
        idx = 0
        for i in self.team_ids:
            for j in self.engine_ids:
                for t in range(1, self.T):
                    if self.x[i,j,t].value() == 1:
                        end_day = min(t + self.work_times[i,j] - 1, self.T-1)
                        schedule[idx] = [self.team_types[i], i, t, end_day, j, self.potential_saved_costs[i,j,t]]
                        idx += 1
                        #print("Team: " + str(self.team_ids[i]) +  " Engine: "+ str(j) + " Starting day: " + str(t) + " End day: " + str(end_day) + " = " + str(self.x[self.team_ids[i],j,t].value()) + "(" + str(self.potential_saved_costs[self.team_ids[i],j,t]) + ")" )
        #print("|----------------|")             
        #print(value(self.model.objective))
        schedule_df = pd.DataFrame.from_dict(schedule, orient='index')
        schedule_df.columns = [
            'team_type',
            'team_id',
            'start_day',
            'end_day',
            'engine_id',
            'cost_saved'
        ]
        schedule_df = schedule_df.sort_values(['team_type', 'team_id', 'start_day']).reset_index(drop=True)
        return schedule_df

    #Purpose: Returns a dataframe which contains the final cost for an examined engine within planning horizon
    def get_engine_costs(self):
        opt_engine_cost = {}
        for j in self.engine_ids:
            total_cost_engine = sum(self.engine_costs[j,t] for t in range(1, self.T+1))
            total_saved_engine = sum(self.potential_saved_costs[i,j,t] * self.x[i,j,t].value()
                                    for t in range(1, self.T)
                                    for i in self.team_ids
                                    )
            opt_engine_cost[j] = [total_cost_engine, total_saved_engine, total_cost_engine - total_saved_engine]
        opt_engine_cost_df = pd.DataFrame.from_dict(opt_engine_cost, orient='index').reset_index()
        opt_engine_cost_df.columns = ['engine_id', 'before_maintenance_cost', 'cost_saved', 'final_engine_cost']
        return opt_engine_cost_df