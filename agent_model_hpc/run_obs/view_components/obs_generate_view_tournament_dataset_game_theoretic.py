#!/usr/bin/env python3
'''

    - generate trials per agents 

'''
import sys, os, getopt
from time import process_time_ns, time_ns
import re
from datetime import datetime

import json
import csv
import pandas as pd
import plotly.graph_objects as go

from run_obs.view_util import View_utility as view_utility
import agent_model.gameforms.agent_topology as rgs

def main(argv):


    
    hpc_config = view_utility().load_hpc_config()
    
    obs_data = initialise_obs_data()
    parse_input(argv, obs_data)
    view_utility().set_basepath(obs_data)
    
    view_utility().eo_ts_o_view_set_obs_data_start(obs_data)
    print(os.path.basename(__file__) + ":: obs_exp_summary: " + obs_data["journal_obs_summary_filename"])
    obs_data_summary = view_utility().retrieve_obs_data_summary(obs_data)
    obs_data["obs_exp"]["obs_subjob_data_path"] = hpc_config["paths"]["observations"] + [obs_data_summary["obs_exp"]["exp_type"], obs_data_summary["obs_exp"]["exp_parent_id"], hpc_config["obs"]["leaf_data"]]
    obs_data["obs_exp"]["sj_count"] = len(obs_data_summary["obs_exp"]["exp_subjobs_list"])
    print(os.path.basename(__file__) + ":: exp_type: " + obs_data_summary["obs_exp"]["exp_type"])
        
    path = os.path.join(obs_data["obs_invocation"]["basepath"], os.sep.join(hpc_config["paths"]["observations"]), obs_data_summary["obs_exp"]["exp_type"], obs_data_summary["obs_exp"]["exp_parent_id"], "view", "ep_o")
    
    
    rgs_object = rgs.agent_topology()
    games = rgs_object.nbs_location_table()
    
    game_theoretic_indices = [1, 2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 22, 23, 29]
    game_theoretic_a0 = ['allc', 'allc', 'allc', 'allc', 'allc', 'alld', 'alld', 'alld', 'alld', 'bully_naive', 'bully_naive', 'bully_naive', 'fictitiousplay', 'fictitiousplay', 'tft']
    game_theoretic_a1 = ['alld', 'bully_naive', 'fictitiousplay', 'tft', 'random', 'bully_naive', 'fictitiousplay', 'tft', 'random', 'fictitiousplay', 'tft', 'random', 'tft', 'random', 'random']
    
    mc_lookup = {
        (0,0) : "CC",
        (0,1) : "CD",
        (1,0) : "DC",
        (1,1) : "DD"
    }
                          
    # tournament running total: agent: reward, mc, %-reward, %-mc
    strategies_tournament_total = { "allc" : [0, 0, 0, 0], "alld" : [0, 0, 0, 0], "bully_naive" : [0, 0, 0, 0], "fictitiousplay" : [0, 0, 0, 0], "tft" : [0, 0, 0, 0], "random" : [0, 0, 0, 0]}
    
    # max scores
    reward_upper_bound_game = 5 * 4 * 1000
    mc_upper_bound_game = 5 * 1000

    reward_upper_bound_tournament = 0
    mc_upper_bound_tournament = 0
    
    
    #for i in range(0, 1):
    for i in range(0, obs_data["obs_exp"]["sj_count"]):
        
        reward_upper_bound_tournament = (i+1) * reward_upper_bound_game
        mc_upper_bound_tournament = (i+1) * mc_upper_bound_game
        
        a0 = []     # list for strategy names agent 0 
        a1 = []     # list for strategy names agent 1
        mc1 = []    # NBS total both location NBS1 and NBS 2
        r0 = []     # reward for agent 0
        r1 = []     # reward for agent 1
        
        outcome_loc_1 = []
        outcome_loc_2 = []
        
        # game running total: agent: reward, mc, %-reward, %-mc
        strategies_game_model = { "allc" : [0, 0, 0, 0], "alld" : [0, 0, 0, 0], "bully_naive" : [0, 0, 0, 0], "fictitiousplay" : [0, 0, 0, 0], "tft" : [0, 0, 0, 0], "random" : [0, 0, 0, 0]}
        
        game = obs_data_summary["obs_exp"]["exp_subjobs"][str(i)]["exp_summary"]["exp_invocation"]["gameform"]
        
        
        file = "view_" + obs_data["obs_exp"]["exp_parent_id"] + "_sj_" + str(i) + "_model_" + game + "_trial_outcomes.csv" 
        data = pd.read_csv(os.path.join(path, file))

        k = 0           # index for strategy name
        j_count = 15    # num valid games in tournament_n^2 
        
        # iterate over all trials in tournament^2
        for j in data.index.tolist():
        
            # filter out duplicates (where agent position is only change, so allc v alld == alld v allc); 
            # also filter out selfplay
            if j in game_theoretic_indices:
                
                # strategy names
                a0.append(game_theoretic_a0[k])
                a1.append(game_theoretic_a1[k])
                
                # total reward from trial for each agent
                r0.append(int(data.iloc[j][4]))
                r1.append(int(data.iloc[j][5]))
                
                # add reward to running total for each agent
                strategies_game_model[game_theoretic_a0[k]][0] += int(data.iloc[j][4])
                strategies_game_model[game_theoretic_a1[k]][0] += int(data.iloc[j][5])

                strategies_tournament_total[game_theoretic_a0[k]][0] += int(data.iloc[j][4])
                strategies_tournament_total[game_theoretic_a1[k]][0] += int(data.iloc[j][5])
                
                # get game's NBS location 1
                game = obs_data_summary["obs_exp"]["exp_subjobs"][str(i)]["exp_summary"]["exp_invocation"]["gameform"]
                outcome_loc_1 = tuple(games[game][0])
                
                # map to dataset with substituted MC location
                mc = int(data.iloc[j][mc_lookup[outcome_loc_1]])

                # if game model has two NBS, then lookup NBS2 location
                # and extract count of that outcome 
                if games[game][1][0] is not None:
                    outcome_loc_2 = tuple(games[game][1])
                    mc2 = int(data.iloc[j][mc_lookup[outcome_loc_2]])
                    
                    # add 2nd count to first
                    mc += mc2
                    
                mc1.append(mc)
                
                strategies_game_model[game_theoretic_a0[k]][1] += mc
                strategies_game_model[game_theoretic_a1[k]][1] += mc
                 
                strategies_tournament_total[game_theoretic_a0[k]][1] += mc
                strategies_tournament_total[game_theoretic_a1[k]][1] += mc

                k = k + 1

        for s in strategies_game_model.items():            
            strategies_game_model[s[0]][2] = strategies_game_model[s[0]][0] / reward_upper_bound_game
            strategies_game_model[s[0]][3] = strategies_game_model[s[0]][1] / mc_upper_bound_game
            
          
        xaxis_trial_count = range(1, j_count + 1)
        data_traces = {
            "A0": pd.Series(a0, index = xaxis_trial_count),
            "A1": pd.Series(a1, index = xaxis_trial_count),
            "R0": pd.Series(r0, index = xaxis_trial_count),
            "R1": pd.Series(r1, index = xaxis_trial_count),
            "MC": pd.Series(mc1, index = xaxis_trial_count),
        }    
        
        df = pd.DataFrame(data_traces)
    
        path2 = os.path.join(obs_data["obs_invocation"]["basepath"], os.sep.join(hpc_config["paths"]["observations"]), obs_data_summary["obs_exp"]["exp_type"], obs_data_summary["obs_exp"]["exp_parent_id"], "view", "ep_o")
        file2 = "view_" + obs_data_summary["obs_exp"]["exp_parent_id"] + "_sj_" + str(i) + "_model_" + game + "_trial_NBS.csv"
        filename = os.path.join(path2, file2)
    
        df.to_csv(filename, encoding='utf-8', index=False)
   
        # write strategies_game_model to files
            
        path3 = os.path.join(os.sep.join(hpc_config["paths"]["observations"]), obs_data_summary["obs_exp"]["exp_type"], obs_data_summary["obs_exp"]["exp_parent_id"], "view", "ep_o")
        file3 = "view_" + obs_data_summary["obs_exp"]["exp_parent_id"] + "_sj_" + str(i) + "_model_" + game + "_trial_NBS_summary.csv"

        with open(os.path.join(obs_data["obs_invocation"]["basepath"], path3, file3), 'w', newline='') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(["Agent", "Reward", "MC", "TR", "MCR"])
            for k, values in strategies_game_model.items():
                write.writerow([k, values[0], values[1], values[2], values[3]])
    
    
    for s in strategies_tournament_total.items():            
        strategies_tournament_total[s[0]][2] = "{:6.4f}".format(strategies_tournament_total[s[0]][0] / reward_upper_bound_tournament)
        strategies_tournament_total[s[0]][3] = "{:6.4f}".format(strategies_tournament_total[s[0]][1] / mc_upper_bound_tournament)

    #print(strategies_tournament_total)
    
    # write strategies_tournament_total to file
        
    path4 = os.path.join(os.sep.join(hpc_config["paths"]["observations"]), obs_data_summary["obs_exp"]["exp_type"], obs_data_summary["obs_exp"]["exp_parent_id"], "view", "ep_o")
    file4 = "view_" + obs_data_summary["obs_exp"]["exp_parent_id"] + "_sj_all_models_trials_NBS_summary_total-RGS.csv"

    with open(os.path.join(obs_data["obs_invocation"]["basepath"], path4, file4), 'w', newline='') as csv_file:
        write = csv.writer(csv_file)
        write.writerow(["Agent", "Reward", "MC", "TR", "MCR"])
        for k, values in strategies_tournament_total.items():
            write.writerow([k, values[0], values[1], values[2], values[3]])
    
    

def parse_input(argv, obs_data):

    try:
        options, args = getopt.getopt(argv, "hj:l:", ["exp_id", "localhost"])
        print(os.path.basename(__file__) + ":: args", options)
    except getopt.GetoptError:
        print(os.path.basename(__file__) + ":: error in input, try -h for help")
        sys.exit(2)
        
    for opt, arg in options:
        if opt == '-h':
            print("usage: " + os.path.basename(__file__) + " \r\n \
            -j <eo_id [__EO_ID__]> | \r\n \
            -l <localhost> boolean (default is false) \r\n \
        ")
        
        elif opt in ('-j', '--pbs_jobstr'):
            obs_data["obs_exp"]["exp_parent_id"] = arg
        
        elif opt in ('-l', '--localhost'):
            if arg == 'true':
                obs_data["obs_invocation"]["localhost"] = True
                
             
    if obs_data["obs_exp"]["exp_parent_id"] == "":
        print(os.path.basename(__file__) + ":: error in input: exp_parent_id is required, use -j __STR__ or try -h for help")
        sys.exit(2)
        
    if not options:
        print(os.path.basename(__file__) + ":: error in input: no options supplied, try -h for help")
        
    else:
        obs_data["obs_invocation"]["obs_args"] = options


def initialise_obs_data():    

    obs_time_start_ns = time_ns()
    
    return {
        "obs_id"                        : str(time_ns()),
        "eo_id"                         :   "",
        "journal_output_filename"       : "",
        "journal_obs_summary_filename"  : "",
        "obs_exp"   : {
            "exp_parent_id"             : "",
            "obs_data_filename_prefix"  : "",
            "sj_count"                  : 0,

        },
        "exp_subjobs"  : {
            "0"                     : {
                "data_files"    : [],
            }
        },
        "obs_invocation"        : {
            "filename"              : __file__,
            "obs_args"              : "",
            "obs_type"              : re.search(r"obs_([A-Za-z_\s]*)", os.path.basename(__file__))[1],
            "obs_time_start_hr"     : datetime.fromtimestamp(obs_time_start_ns / 1E9).strftime("%d%m%Y-%H%M%S"),
            "obs_time_end_hr"       : "",
            "obs_time_start_ns"     : obs_time_start_ns,
            "obs_time_end_ns"       : 0,
            "process_start_ns"      : process_time_ns(),
            "process_end_ns"        : 0,
            "localhost"             : False,
            "home"				    : "",
            "basepath"				: "",
        },

        
    }







if __name__ == '__main__':
    main(sys.argv[1:]) 
    