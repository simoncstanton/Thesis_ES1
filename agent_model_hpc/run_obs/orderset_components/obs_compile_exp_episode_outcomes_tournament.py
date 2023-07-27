#!/usr/bin/env python3
# file: obs_compile_exp_episode_outcomes_tournament.py
'''
    obs_compile_exp_episode_outcomes_tournament
    
    usage: python -m obs_compile_exp_episode_outcomes_tournament -j __STR__ -z false -l true
        
        - option -j __EXP_ID__
        - option -z only required if setting to true (default is false). 
            - flips flag to write data as compressed files (json.gz and csv.gz).
        - option -l (localhost, ie computer other than PBS/HPC) 
            - only required if setting to true (default is false)
            - used for setting basepath
            
        nominative key
            eo_ep_o   [experiment observation episode outcomes]
        
        
        Activity
        
        - analyses a set of episodes (minimum == 1) across a set of subjobs (minimum == 1)
            
            - for each subjob
                - for each trial (matching of two strategies)    
                    - extract final-timestep outcome count        
                        - write to __EXP_ID__/data/ep_o/obs_\__EXP_ID__\_sj_\__SJ__\_ep_o_terminal_e_count.csv
                            - for each outcome write total for that outcome, per match (so four lines, of #match entries each)
                                cc: mutual cooperation
                                cd: cooperate-defect
                                dc: defect-cooperate
                                dd: mutual defection
                  

                

'''

import sys, os, getopt

from time import process_time_ns, time_ns
from datetime import datetime

import re
import math
from natsort import natsorted

from run_obs.obs_util import Obs_utility as obs_utility

def main(argv):
    
    hpc_config = obs_utility().load_hpc_config()
    
    obs_data = initialise_obs_data()
    parse_input(argv, obs_data)
    obs_utility().set_basepath(obs_data)
    
    obs_utility().eo_ep_o_set_obs_data_start(obs_data)    
    obs_data_summary = obs_utility().retrieve_obs_data_summary(obs_data)
    
    obs_utility().make_ep_o_job_paths(obs_data, obs_data_summary)
        
        
    tournament_strategy_count = len(obs_data_summary["obs_exp"]["exp_strategy_list"])
    print("tournament_strategy_count ", tournament_strategy_count)
    tournament_trial_count = pow(tournament_strategy_count, 2)
        
    # result_set_data = {}
    # leaf_dir = hpc_config["exp_data_leaf_dirs"]["cumulative_outcome_history"]
    

    # result_set_data = obs_utility().fetch_all_subjobs_result_set_data_tournament(obs_data, obs_data_summary, leaf_dir, tournament_trial_count)
    # print(len(result_set_data), len(result_set_data[0]), list(result_set_data[0].keys())[0])

            
    # label = "129249_143_24_2_dd"        
    # print(label, result_set_data[143][label])  






    result_set_data = {}
    #leaf_dir = hpc_config["exp_data_leaf_dirs"]["reward_history"]
    leaf_dir = hpc_config["exp_data_leaf_dirs"]["cumulative_outcome_history"]
    
    
    result_set_data = obs_utility().fetch_all_subjobs_result_set_data_tournament(obs_data, obs_data_summary, leaf_dir, tournament_trial_count)
    print(len(result_set_data), len(result_set_data[0]), list(result_set_data[0].keys())[0])


    




    
    create_obs_data_exp_subjob_dict(obs_data, len(obs_data_summary["obs_exp"]["exp_subjobs_list"]))
    

    
    
    
    
    '''
        ep_o_assemble_terminal_e_count()
        
    '''
    obs_utility().ep_o_assemble_terminal_e_count_tournament(result_set_data, obs_data, obs_data_summary) 







    '''
        re-iterate over output data to index to exp_summary strategy field and update current obs_data
    '''

    #for sj, item in sj_data_mean_terminal_distribution.items():
    #    obs_data["obs_exp"]["exp_subjobs"][sj]["strategy"] = obs_data_summary["obs_exp"]["exp_subjobs"][str(sj)]["exp_summary"]["exp_invocation"]["strategy"]











    obs_utility().eo_ep_o_set_obs_data_end(obs_data, obs_data_summary)
    obs_utility().eo_ep_o_write_obs_data_summary(obs_data, obs_data_summary)
    obs_utility().eo_ep_o_write_obs_journal(obs_data, obs_data_summary)
    
    
            
            


def parse_input(argv, obs_data):

    try:
        options, args = getopt.getopt(argv, "hj:z:l:", ["exp_id", "compress_writes", "localhost"])
        print(os.path.basename(__file__) + ":: args", options)
    except getopt.GetoptError:
        print(os.path.basename(__file__) + ":: error in input, try -h for help")
        sys.exit(2)
        
    for opt, arg in options:
        if opt == '-h':
            print("usage: " + os.path.basename(__file__) + " \r\n \
            -j <eo_id [__EO_ID__]> | \r\n \
            -z <compress_writes> boolean (default is true) \r\n \
            -l <localhost> boolean (default is false) \r\n \
        ")
        
        elif opt in ('-j', '--pbs_jobstr'):
            obs_data["obs_exp"]["exp_parent_id"] = arg
            
        elif opt in ('-z', '--compress_writes'):
            if arg == 'true':
                obs_data["obs_invocation"]["compress_writes"] = True
        
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


def create_obs_data_exp_subjob_dict(obs_data, sj_count):
    for sj in range(0, sj_count):
        obs_data["obs_exp"]["exp_subjobs"][str(sj)] = {
            "gameform"      : "",
            "data_files"    : {},
        }
        
def initialise_obs_data():    

    obs_time_start_ns = time_ns()
    
    return {
        "obs_id"                    : str(time_ns()),
        "eo_id"                     :   "",
        "journal_output_filename"   : "",
        "obs_exp"   : {
            "exp_parent_id"             : "",
            "obs_data_filename_prefix"  : "",
            "sj_count"                  : 0,
            "strategy_count"            : 0,
            "obs_subjob_data_path"      : "",
            "exp_subjobs"  : {
                "0"                     : {
                    "gameform"      : "",
                    "data_files"    : [],
                }
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
            "compress_writes"       : False,
            "localhost"             : False,
            "home"				    : "",
            "basepath"				: "",
        }
        
    }

    
        
if __name__ == '__main__':
    main(sys.argv[1:])        