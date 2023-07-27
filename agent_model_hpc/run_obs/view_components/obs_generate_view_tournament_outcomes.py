#!/usr/bin/env python3
'''

    - generate trials per agents 

'''
import sys, os, getopt
from time import process_time_ns, time_ns
import re
from datetime import datetime

import json
import pandas as pd
import plotly.graph_objects as go

from run_obs.view_util import View_utility as view_utility


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
    
    

    
    for i in range(0, obs_data["obs_exp"]["sj_count"]):
        game = obs_data_summary["obs_exp"]["exp_subjobs"][str(i)]["exp_summary"]["exp_invocation"]["gameform"]
        #print(game)
        
        # want this
        #   basepath\results\observations\tournament\001\data\ep_o\obs_001_sj_0_ep_o_terminal_e_count.csv
        o_path = os.path.join(os.sep.join(obs_data["obs_exp"]["obs_subjob_data_path"]), "ep_o")
        o_file = "obs_" + obs_data["obs_exp"]["exp_parent_id"] + "_sj_" + str(i) + "_ep_o_" + "terminal_e_count.csv"
        data_outcomes = view_utility().fetch_data(o_path, o_file, obs_data)
        
        r_path = os.path.join(os.sep.join(obs_data["obs_exp"]["obs_subjob_data_path"]), "ep_r")
        r_file = "obs_" + obs_data["obs_exp"]["exp_parent_id"] + "_sj_" + str(i) + "_ep_r_terminal_sums_tournament.csv"  
        data_rewards = view_utility().fetch_data(r_path, r_file, obs_data)[0]
        

        import ast
        
        data_reward_0 = []
        data_reward_1 = []
        
        for d in data_rewards:
            agents = ast.literal_eval(d)
            data_reward_0.append(agents["0"])
            data_reward_1.append(agents["1"])

        xaxis_trial_count = range(1, len(data_outcomes[0])+1)
        
        
        data_traces = {
            'CC': pd.Series(data_outcomes[0], index = xaxis_trial_count),
            'CD': pd.Series(data_outcomes[1], index = xaxis_trial_count),
            'DC': pd.Series(data_outcomes[2], index = xaxis_trial_count),
            'DD': pd.Series(data_outcomes[3], index = xaxis_trial_count),
            '0': pd.Series(data_reward_0, index = xaxis_trial_count),
            '1': pd.Series(data_reward_1, index = xaxis_trial_count),
        }

        df = pd.DataFrame(data_traces)
    
        path = os.path.join(obs_data["obs_invocation"]["basepath"], os.sep.join(hpc_config["paths"]["observations"]), obs_data_summary["obs_exp"]["exp_type"], obs_data_summary["obs_exp"]["exp_parent_id"], "view", "ep_o")
        file = "view_" + obs_data_summary["obs_exp"]["exp_parent_id"] + "_sj_" + str(i) + "_model_" + game + "_trial_outcomes.csv"
        filename = os.path.join(path, file)
    
        df.to_csv(filename, encoding='utf-8', index=False)
        
        


        
        
        
        
        
    #view_utility().eo_ts_o_view_set_obs_data_end(obs_data, obs_data_summary)
    #view_utility().eo_ts_o_view_write_obs_data_summary(obs_data, obs_data_summary)
    #view_utility().eo_ts_o_view_write_obs_journal(obs_data, obs_data_summary)




  



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
    