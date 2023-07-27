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
    
    bandit_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,57,58,59,60,61,62,63,64,65,66,67,73,74,75,76,77,78,79,80,81,82,83,84,91,92,93,94,95,96,97,98,99,100,101,109,110,111,112,113,114,115,116,117,118,127,128,129,130,131,132,133,134,135,145,146,147,148,149,150,151,152,163,164,165,166,167,168,169,181,182,183,184,185,186,199,200,201,202,203,217,218,219,220,235,236,237,253,254,271]
    bandit_a0 = ['bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_inc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_noninc_softmax_ap_2ed','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_pursuit_sav','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_reinfcomp','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_inc_softmax','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sav_noninc_softmax','bandit_sav_noninc_softmax','bandit_sav_noninc_softmax','bandit_sav_noninc_softmax','bandit_sav_noninc_softmax','bandit_sav_noninc_softmax','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_direct','bandit_sl_direct','bandit_sl_direct','bandit_sl_direct','bandit_sl_direct','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lri','bandit_sl_la_lri','bandit_sl_la_lri','bandit_sl_la_lri','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_sl_la_lrp','bandit_sl_la_lrp','bandit_sl_la_lrp','bandit_sl_la_lrp','bandit_wa','bandit_wa','bandit_wa','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_optimistic_greedy','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax','bandit_wa_softmax_ap_2ed']
    bandit_a1 = ['bandit_noninc_softmax_ap_2ed','bandit_pursuit_sav','bandit_reinfcomp','bandit_sav_inc','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_softmax','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_pursuit_sav','bandit_reinfcomp','bandit_sav_inc','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_softmax','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_reinfcomp','bandit_sav_inc','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_softmax','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sav_inc','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_softmax','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sav_inc_optimistic_greedy','bandit_sav_inc_softmax','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sav_inc_softmax','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sav_noninc','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sav_noninc_softmax','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sl_direct','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sl_la_lri','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_sl_la_lrp','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_wa','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_wa_optimistic_greedy','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_wa_softmax','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_wa_softmax_ap_2ed','bandit_wa_ucb','bandit_wa_ucb']
    
    mc_lookup = {
        (0,0) : "CC",
        (0,1) : "CD",
        (1,0) : "DC",
        (1,1) : "DD"
    }
                          
    # tournament running total: agent: reward, mc, %-reward, %-mc
    strategies_tournament_total = {"bandit_inc_softmax_ap_2ed" : [0, 0, 0, 0], "bandit_noninc_softmax_ap_2ed" : [0, 0, 0, 0], "bandit_pursuit_sav" : [0, 0, 0, 0], "bandit_reinfcomp" : [0, 0, 0, 0], "bandit_sav_inc" : [0, 0, 0, 0], "bandit_sav_inc_optimistic_greedy" : [0, 0, 0, 0], "bandit_sav_inc_softmax" : [0, 0, 0, 0], "bandit_sav_noninc" : [0, 0, 0, 0], "bandit_sav_noninc_softmax" : [0, 0, 0, 0], "bandit_sl_direct" : [0, 0, 0, 0], "bandit_sl_la_lri" : [0, 0, 0, 0], "bandit_sl_la_lrp" : [0, 0, 0, 0], "bandit_wa" : [0, 0, 0, 0], "bandit_wa_optimistic_greedy" : [0, 0, 0, 0], "bandit_wa_softmax" : [0, 0, 0, 0], "bandit_wa_softmax_ap_2ed" : [0, 0, 0, 0], "bandit_wa_ucb" : [0, 0, 0, 0]}
    
    # max scores
    reward_upper_bound_game = 16 * 4 * 1000
    mc_upper_bound_game = 16 * 1000

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
        strategies_game_model = {"bandit_inc_softmax_ap_2ed" : [0, 0, 0, 0], "bandit_noninc_softmax_ap_2ed" : [0, 0, 0, 0], "bandit_pursuit_sav" : [0, 0, 0, 0], "bandit_reinfcomp" : [0, 0, 0, 0], "bandit_sav_inc" : [0, 0, 0, 0], "bandit_sav_inc_optimistic_greedy" : [0, 0, 0, 0], "bandit_sav_inc_softmax" : [0, 0, 0, 0], "bandit_sav_noninc" : [0, 0, 0, 0], "bandit_sav_noninc_softmax" : [0, 0, 0, 0], "bandit_sl_direct" : [0, 0, 0, 0], "bandit_sl_la_lri" : [0, 0, 0, 0], "bandit_sl_la_lrp" : [0, 0, 0, 0], "bandit_wa" : [0, 0, 0, 0], "bandit_wa_optimistic_greedy" : [0, 0, 0, 0], "bandit_wa_softmax" : [0, 0, 0, 0], "bandit_wa_softmax_ap_2ed" : [0, 0, 0, 0], "bandit_wa_ucb" : [0, 0, 0, 0]}
        
        game = obs_data_summary["obs_exp"]["exp_subjobs"][str(i)]["exp_summary"]["exp_invocation"]["gameform"]
        
        
        file = "view_" + obs_data["obs_exp"]["exp_parent_id"] + "_sj_" + str(i) + "_model_" + game + "_trial_outcomes.csv" 
        data = pd.read_csv(os.path.join(path, file))

        k = 0               # index for strategy name
        j_count = 136       # num valid games in tournament_n^2 
        
        # iterate over all trials in tournament^2
        for j in data.index.tolist():
        
            # filter out duplicates (where agent position is only change, so allc v alld == alld v allc); 
            # also filter out selfplay
            if j in bandit_indices:
                
                # strategy names
                a0.append(bandit_a0[k])
                a1.append(bandit_a1[k])
                
                # total reward from trial for each agent
                r0.append(int(data.iloc[j][4]))
                r1.append(int(data.iloc[j][5]))
                
                # add reward to running total for each agent
                strategies_game_model[bandit_a0[k]][0] += int(data.iloc[j][4])
                strategies_game_model[bandit_a1[k]][0] += int(data.iloc[j][5])

                strategies_tournament_total[bandit_a0[k]][0] += int(data.iloc[j][4])
                strategies_tournament_total[bandit_a1[k]][0] += int(data.iloc[j][5])
                
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
                
                strategies_game_model[bandit_a0[k]][1] += mc
                strategies_game_model[bandit_a1[k]][1] += mc
                 
                strategies_tournament_total[bandit_a0[k]][1] += mc
                strategies_tournament_total[bandit_a1[k]][1] += mc

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
            write.writerow(["Agent", "Reward", "MC", "%-Reward", "%-MC"])
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
        write.writerow(["Agent", "Reward", "MC", "%-Reward", "%-MC"])
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
    