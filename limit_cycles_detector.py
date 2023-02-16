import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import itertools

def convert_line_patches(x,y,size):
    x_old, y_old = x,y
    

    #create intermediate points between each x and y so that it can look as a lime plot either on a matrix
    points_in = 10 #nummber of intermediate points between to points
    x_interpolated, y_interpolated = [], []
    for _ in range(1, len(x) - 1):
    
        #slope between the 2 points
        x0, x1, y0, y1 = x[_ - 1], x[_], y[_ - 1], y[_]
        if x1 != x0:
            slope = (y1 - y0) / (x1 - x0)
            intercept = y0 - (slope * x0)
            
            #generate points
            new_x = list(np.arange(x0,x1,(x1-x0)/points_in) + (x1-x0)/(points_in * 2)) #last part to center it
            new_y = []
            for nx in new_x: new_y.append(slope * nx + intercept)

            x_interpolated = x_interpolated + new_x
            y_interpolated = y_interpolated + new_y

    x = x + x_interpolated
    y = y + y_interpolated
    
    #plt.plot(x,y, label = 'new', c = "navy")
    #plt.plot(x_old, y_old, label = 'old', c = "salmon")
    #plt.legend()
    #plt.show()

    #create matrix from the line x-y
    matrix = np.zeros((size,size))
    max_x, min_x, max_y, min_y = max(x), min(x), max(y), min(y)
    if max_x != min_x and  max_y != min_y:
        for i in range(len(x)):
            xi = int ( size * ( x[i] - min_x ) / ( max_x - min_x) )
            yi = int ( size * ( y[i] - min_y ) / ( max_y - min_y) )

            xi = min(max(xi,0), size - 1)
            yi = min(max(yi,0), size - 1)

            matrix[xi, yi] = 1

    return matrix

def detect_cycles(matrix):

    def test_cell(to_explore, size_hole, border_hole, explored_xy):

        def testing_operation(border_hole):
            #print(x_new, y_new)
            #print("matrice: ", matrix[x_new, y_new] == 0)
            if matrix[x_new, y_new] == 0: #check if it is not a place with the line
                #print(y_new < (size - 1))
                #print(x_new == 0, x_new == (size - 1), y_new == 0, y_new == (size - 1))
                if x_new == 0 or x_new == (size - 1) or y_new == 0 or y_new == (size - 1): #check if in the borders
                    #print("border_hole True")
                    border_hole = True
                else:
                    point_to_append = [ x_new, y_new ]
                    if ( point_to_append not in to_explore ) and ( point_to_append not in explored_xy ): #check if already been there
                        to_explore.append(point_to_append)
            
            return border_hole
        

        target_cell = to_explore[0]
        #print(target_cell)
        xt, yt = target_cell[0], target_cell[1]
        #check cell on the right
        x_new, y_new = xt + 1, yt
        border_hole = testing_operation(border_hole)
        #check cell on the left
        x_new, y_new = xt - 1, yt
        border_hole = testing_operation(border_hole)
        #check cell upwards
        x_new, y_new = xt, yt + 1
        border_hole = testing_operation(border_hole)
        #check cell downwards
        x_new, y_new = xt, yt - 1
        border_hole = testing_operation(border_hole)

        #finish the exploration
        test_turn = True
        size_hole += 1
        to_explore = to_explore[1:]
        explored.append(target_cell)
        explored_xy.append(target_cell)

        
        return to_explore, explored, size_hole, border_hole, explored_xy

    size = len(matrix)
    control_matrix = np.zeros((size, size))
    sizes_holes = []
    explored = []

    #test each cell
    for x in range(size):
        for y in range(size):
            
            
            test_line = matrix[x,y] == 1 #if is full, do not count
            test_border = x == 0 or y == 0 or x == (size-1) or y == (size-1) #if on border, do not count
            test_already_checked = [x,y] in explored #if the area is already check, do not count

            #print("phase 1: ", x,y, not (test_line or test_border or test_already_checked))
            
            
            if not (test_line or test_border or test_already_checked):

                size_hole = 0 #start from an hole of size 0 (when exploring the first, increasing of one unit)
                to_explore = [[x,y]] #start from exploring central point
                explored_xy = [] #points explored in this iteration

                #print("********* phase 2: ", x,y, " ***********")

                while len(to_explore) > 0:

                    #print("phase 3: ", x,y, "size hole: ", size_hole)
                    #print("phase 3 --> to explore: ", to_explore)
                    
                    test_turn = False
                    border_hole = False #to see if it is a border hole, so not to count
                    if len(to_explore) > 0: to_explore, explored, size_hole, border_hole, explored_xy = test_cell(to_explore, size_hole, border_hole, explored_xy)
                    
                    if border_hole: 
                        test_turn = True
                        size_hole = 0
                        to_explore = []
                        #print("phase 3 - border_hole: ", x,y)
                    
                    if len(to_explore) == 0 or test_turn:
                        
                        if size_hole > 0: sizes_holes.append(size_hole)

    max_size_holes = 0             
    if len(sizes_holes) > 0: max_size_holes = max(sizes_holes)


    return max_size_holes

def space_exploration(function, boundaries, types, decimals, df_results, n_tests, sim_repetition, name_file, threshold_expl):

    def init_results_list(boundaries):
        return [[] for _ in range(len(boundaries) + 1)]

    #read a list of 2-elements list and the tyoe of variable and returns a list of uniform (int or float) numbers
    def generate_random_numbers(boundaries, types):
        random_values = []
        for _ in range(len(boundaries)):
            sublist, type_ = boundaries[_], types[_]
            lower_bound, upper_bound = sublist[0], sublist[1]
            decimals_ = decimals[_]
            if type_ == 'int':
                random_values.append(round(random.randint(lower_bound, upper_bound),decimals_))
            else:
                random_values.append(round(random.uniform(lower_bound, upper_bound),decimals_))
  
        return random_values

    def exp_perform(function, pars, sim_repetition):
        results = []
        for sim in range(sim_repetition):
            
            try:
                new_result = function(*pars)
                #result = 1 if function(*pars) > 0.2 else 0
                results.append(new_result) #it should return the ration of circle on the total space of the matrix
            except:
                print("Error at line 141, pars: ", pars)
                #print("Type pars: ", type(pars))
                break
        
        return np.mean(results)

    def result_store(pars, mean_results, results):
        
        for _ in range(len(pars)): results[_].append(pars[_])
        
        results[len(results) - 1].append(mean_results) #results are always at the end

        return results

    def neighbourhood_exploration(function, pars, boundaries, results, explorations, decimals, types, threshold_expl, df_results):

        def generate_points_to_explore(start_point, boundaries, decimals, types, threshold_expl):
            #granularity = 100
            #delta = [0 if b[1] == b[0] else abs(b[1] - b[0]) / granularity if t == 'float' else 1 for b, t in zip(boundaries, types)]
            delta = [0 if b[1] == b[0] else 1 * 10**(-d) if t == 'float' else 1 for b, t, d in zip(boundaries, types, decimals)]
            to_explore = []
            variations = [-1 for b in boundaries]
            repetitions = len(start_point) ** 3
            while len(to_explore) < repetitions:
                
                #change variations
                if variations[0] <= 1: 
                    to_explore.append([d * v + sp for d,v,sp in zip(delta,variations,start_point)])
                    variations[0] += 1
                else:
                    variations[0] = -1
                    i, flag2 = 1, True
                    while flag2:
                        variations[i] += 1
                        if variations[i] <= 1: 
                            flag2 = False
                        else:
                            variations[i] = -1
                            i += 1
                
                try: 
                    to_explore.remove(start_point) #at the end remove the starting point
                except:
                    pass

            #remove the ones beyond the space boundaries
            old_to_explore = to_explore
            flags = []
            for e in to_explore:
                flag_remove = False
                for _ in range(len(e)):
                    if e[_] < boundaries[_][0] or e[_] > boundaries[_][1]: 
                        flag_remove = True
                flags.append(flag_remove)
            
            to_explore = [e for e,f in zip(to_explore,flags) if not f]

            #round each element of the list e (which made to_explore) by the correct number of decimals (in list decimals)
            new_to_explore = []
            for e in to_explore:
                new_element = []
                for _ in range(len(to_explore[0])): 
                    new_element.append(round(e[_], decimals[_]))
                new_to_explore.append(new_element)
            to_explore = new_to_explore
            #to_explore = [[round(e[0], decimals[0]), round(e[1], decimals[1])] for e in to_explore]
            return to_explore

        list_explored, flag, explore_from, to_explore = [], True, [], []
        explorations += 1
        i = 0
        #initialize the loop with the very first point, which I am sure is okay
        explore_from = generate_points_to_explore(pars, boundaries, decimals, types, threshold_expl) #create new points to explore from a central point
        while flag:
            i += 1
            # 1 - check that the point to explored were not explored yet, otherwise remove the
            explore_from = [e for e in explore_from if e not in list_explored]
            # 2 - generate all points to explore (closeby the last explored)
            if len(explore_from) == 0:
                flag = False
                break
            explore_from_0 = explore_from[0] #get the point from which the exploration starts
            #print(len(explore_from))

            # 3 - remove from the list of the elements to explored
            explore_from.remove(explore_from_0)
            list_explored.append(explore_from_0)

            # 4 - take the first point and generate a result
            mean_results = exp_perform(function, explore_from_0, sim_repetition)

            # 5 - add result to list of results
            results = result_store(explore_from_0, mean_results, results)

            # 6 - check if result > 0 or not:
            if mean_results > threshold_expl:

                #if mr > 0, add the point explored to the one "to explore from", used earlier to get new points to explore
                new_to_explore = generate_points_to_explore(explore_from_0, boundaries, decimals, types, threshold_expl) #create new points to explore from a central point
                explore_from = explore_from + new_to_explore
                #remove duplicates
                explore_from.sort()
                explore_from = list(k for k,_ in itertools.groupby(explore_from))

            # 7 - finally, if no elements in to_explore, leave the loop
            if len(explore_from) == 0: flag = False

            # 8 - Once in a while, save the results of the exploration
            if i % 10 == 0 and i > 1:
                df_results = result_export(results, df_results)
                results = init_results_list(boundaries) #initialize again so no storing same results

        return explorations, results

    def result_export(results, df_results):

        new_df = pd.DataFrame(columns=df_results.columns)
        for _ in range(len(results)): new_df[new_df.columns[_]] = results[_]
        
        df_results = pd.concat([df_results, new_df], axis=0)
        df_results.to_excel(name_file, index = False)

        return df_results

        
    #CODE STARTS FROM HERE
    results = init_results_list(boundaries)
    explorations = 0
    for n in range(n_tests):
         
        #generate a random lists of elements from the boundaries
        pars = generate_random_numbers(boundaries, types)
        print('simulations: ' + str(n + 1) + ' - explorations: ' + str(explorations), ' - pars: ', pars, end = '\r')

        #perform the experiments
        mean_results = exp_perform(function, pars, sim_repetition)

        #store result
        results = result_store(pars, mean_results, results)
        
        #explore the neighboorhood of the point if result is good
        if mean_results > threshold_expl: 
            print('EXPLORATION!!! On simulation: ' + str(n + 1) + ' - explorations: ' + str(explorations), ' - pars: ', pars, end = '\r')
            explorations, results = neighbourhood_exploration(function, pars, boundaries, results, explorations, decimals, types, threshold_expl, df_results)
    
        if n % 10 and n > 0 == 0: 
            df_results = result_export(results, df_results)
            results = init_results_list(boundaries) #initialize again so no storing same results
    
    #store the results
    df_results = result_export(results, df_results)
    return df_results 
    
def space_size(boundaries, types, decimals, n_tests):

    combinations = 1
    for _ in range(len(boundaries)):
        step = 1 if decimals[_] == 0 else 10 ** (-1 * decimals[_])
        combinations *= max(len(np.arange(boundaries[_][0],boundaries[_][1],step)),1)

    pct_space_explored = round(100 * n_tests / combinations, 3)
    print("The space exploration setting has", combinations, "possibile parameterss combinations. You are testing", pct_space_explored, "% of the parameter space.")