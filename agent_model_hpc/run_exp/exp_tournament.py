#!/usr/bin/env python3
'''
Tournament

if run locally, runs a strategy through a supplied gameform; if run from PBS job_array, runs over all topology gameforms

usage: python -m run_exp.exp_tournament -j __STR__ -w __STR__ -g pd -r scalar -e 10 -t 100 -p na -k true -z false -c false -l true 
    
    localhost example: 
        python -m  run_exp.exp_tournament -j '1111.localhost' -w 'localhost' -s allc  -g pd -r scalar -e 10 -t 100 -p 'na' -k true -z false -c false -l true -m true -s true -b true -v false -i 1
    or, 
        python -m  run_exp.exp_tournament -j '1111.localhost' -w 'localhost' -s actor_critic_1ed  -g pd -r scalar -e 10 -t 100 -p 'alpha=0.1:gamma=0.2' -k true -z false -c false -l true -m true -s true -b true  -v false -i 1
    
    - option -k only required if setting to true (default is false). 
        - flips flag to write agent and strategy state - can use when running very large jobs and state may enter the gigabyte + range.

    - option -z only required if setting to false (default is true). 
        - flips flag to write data as compressed files (json.gz and csv.gz).
        
    - option -c only required if setting to false (default is true). 
        - flips flag to write a timestep interval update to a central log file: obs_exp/heartbeat/agent_model.heartbeat
        - if true, on a job_array then IO load may result, on single job, only one process is running so is (should be) ok.
        - if false, will write per-process hearbeat log to obs_exp/heartbeat/__PBS_PARENT_JOBID__/__PBS_JOBID__.heartbeat
    
    - option -l (localhost, ie computer other than PBS/HPC) 
        - only required if setting to true (default is false). 
        - flips flag to write directly to experiments directory (localhost), 
            - or to __PBS_SCRATCH_DIR__ (location which is effectively mounted on execution node on PBS/HPC/cluster)
            - which is moved to experiments dir by __PBS_SCRIPT__ after completion, by PBS.
        - setting to true renders -w option redundant.
        - setting to true implies no PBS supplied __PBS_JOBID__ (-j option)
            - use format '__INT__.localhost' where __INT__ is some manually incremented number eg: '001'
        - setting to true writes the experiments directory directly to exp_data["job_parameters"]["job_dir"]
            - otherwise exp_data["job_parameters"]["job_dir"] contains path to -w __PBS_SCRATCH_DIR__
            
    - option -m sets mode [IGNORE - TODO - I think this is deprecated/not applicable for all but spacewalk experiment type - leaving note here for now until sure (300821)]
        mode is type of exp to run in given gameform. eg: symmetric_selfplay [see below, repurposing to set input parameter flag [TODO]
            [20/09/2021 - Deprecating -m option. It is not used as tournaments are not asymmetric to start with. 
                Leaving -m option in though, to use as a flag to indicate that parameters for a strategy are to be read from the strategy_config file (default), or from the line that contains the strategy name in the strategy_set file. [TODO]
        
    - option -s sets flag to write every timestep in every episode to file (default=FALSE), or only write terminal episode values (TRUE)
    
    - option -v (verbose) suppress various status messages from being sent to std_out. Default is False.
    
    - option -h (heartbeat_interval) set frequency of update to heartbeat log file. Used mostly for tail during execution.
        value is number of powers less than timestep to write heartbeat message, so 0 will write only on timestep 0 for any number of timesteps, but 1 will write every 100 timesteps for total timesteps in episode of 1000.
        Default is 0 (for tournament exp_type)
 
    - option -b (write timestep map). set flag (default is true) to write single episode timestep map of outcomes.
    
    - option -d write verbose heartbeat (default is False)
        
'''

import os, sys, getopt
import re

from time import process_time_ns, time_ns
from datetime import datetime

import math

'''
    agent_model_hpc modules
'''
import agent_model.agent as agent
import agent_model.trial as trial

from agent_model.utility import Utility as utility
from run_exp.exp_util import Exp_utility as exp_utility




def main(argv):
    
    exp_data = initialise_exp_data()
    
    parse_input(exp_data, argv)
    exp_utility().set_basepath(exp_data)
    
    exp_utility().set_exp_data_start(exp_data)
    
    # reset/override heartbeat_interval - was set in last method call. otherwise heartbeat file grows very large. A value of 0 is equal to episode length 
    exp_data["job_parameters"]["heartbeat_interval"] = math.pow(10, int(math.log10(exp_data["exp_invocation"]["timesteps"])) - exp_data["exp_invocation"]["heartbeat_interval"])
    
    
    exp_utility().make_exp_job_paths(exp_data)
    
    ''' 
       
    '''
    matrix = utility().retrieve_matrix(exp_data["exp_invocation"]["gameform"], exp_data["exp_invocation"]["reward_type"])
    strategy_set = exp_utility().load_tournament_strategy_set(exp_data["exp_invocation"]["strategy_set_input"])

    strategy_set_0 = []
    strategy_set_1 = []

    for f in strategy_set.keys():
        strategy_set_0.append(f)
        strategy_set_1.append(f)
    
    exp_data["exp_invocation"]["strategy_list"] = strategy_set
    
    strategy_count = len(strategy_set)    
    trials_in_round = strategy_count ** 2

    tournament_trial_id = 0
    
    for strategy_0 in strategy_set_0:
    
        for strategy_1 in strategy_set_1:
            '''
                assemble the parameters for each agent's strategy
                    - agent context is same gameform (matrix) for both agents
                    - load default strategy parameter values (memory_depth, alpha, gamma etc)
                    
            '''
            agent_0_strategy = strategy_0
            agent_1_strategy = strategy_1
            
            strategy_parameters_0 = exp_utility().load_tournament_strategy_parameters(exp_data, strategy_0, matrix)
            strategy_parameters_1 = exp_utility().load_tournament_strategy_parameters(exp_data, strategy_1, matrix)
            
            set_exp_data_tournament_trial_entry(exp_data, tournament_trial_id, agent_0_strategy, agent_1_strategy, strategy_parameters_0, strategy_parameters_1)
            
            exp_tournament_round(exp_data, matrix, strategy_0, strategy_1, strategy_parameters_0, strategy_parameters_1, tournament_trial_id)
            
            tournament_trial_id += 1
            

    
    ''' 
        
    ''' 
    
    exp_utility().set_exp_data_end(exp_data)
    
    exp_utility().write_exp_data_summary(exp_data)
    exp_utility().write_exp_journal_entry(exp_data)


def set_exp_data_tournament_trial_entry(exp_data, tournament_trial_id, agent_0_strategy, agent_1_strategy, strategy_parameters_0, strategy_parameters_1):
    trial_entry = {
        "exp_tournament_trial_id"          : tournament_trial_id,
        "agents"          : {
            "agent_0"             : { 
                "strategy"          : agent_0_strategy,
                "strategy_parameters" : strategy_parameters_0,
            },
            "agent_1"             : {
                "strategy"          : agent_1_strategy,
                "strategy_parameters" : strategy_parameters_1,
            },
        },
    }
    
    exp_data["exp_tournament_trials"][tournament_trial_id] = trial_entry
    

                        


def exp_tournament_round(exp_data, matrix, strategy_0, strategy_1, strategy_parameters_0, strategy_parameters_1, tournament_trial_id):
    
    exp_utility().make_tournament_trial_folder(exp_data, tournament_trial_id)
    
    if exp_data["exp_invocation"]['verbose']:
        print(os.path.basename(__file__) + ":: tournament_trial_id: ", tournament_trial_id)
    
    ''' 
        - temporary episode and timestep local variables for easy reading. 
            - the values (stored in exp_data) are immutable for the course of the experiment
        
    '''
    
    timesteps = exp_data["exp_invocation"]["timesteps"]
    episodes = exp_data["exp_invocation"]["episodes"]
    
    # for summing episode reward total for each agent; only create if beng used to save memory
    if exp_data["exp_invocation"]["sum_episodes"]:
        agents_reward_episode_totals = {"0" : [0] * episodes, "1" : [0] * episodes}
        sum_episode_outcomes_cumulative = {"cc": [0] * episodes, "cd": [0] * episodes, "dc": [0] * episodes, "dd": [0] * episodes}
       
    ''' 
        -------------------------------------------------------
        start all episodes
        
        # write heartbeat start regardelss of option verbose_heartbeat
        
    '''
    
    exp_utility().heartbeat_start_tournament(exp_data, tournament_trial_id)
    
    for e in range(episodes):
        
        if exp_data["exp_invocation"]['verbose']:
            print(os.path.basename(__file__) + ":: episode: ", e)
        
        '''
            - episode and timestep collections
            
        '''
        
        if not exp_data["exp_invocation"]["sum_episodes"]:
            single_episode_outcomes_cumulative  = {"cc": [], "cd": [], "dc": [], "dd": []}
        
        if exp_data["exp_invocation"]["write_ts_map"]:
            single_episode_outcomes_timestep_map   = {"cc": [0] * timesteps, "cd": [0] * timesteps, "dc": [0] * timesteps, "dd": [0] * timesteps}
        
        
        '''
            instantiate agents
                - Re-instantiate agents at each episode start to effectively clear agent and strategy state.
                    ok for 'static' experiments, ie episodes must have exact same agent/strategy profile,
                    otherwise, can instantiate outside the episode loop, and handle state deliberately (which will be useful in later experiments)
                    
                - agents dynamically load strategy class with default values from strategy_config["default_strategy_parameters"] dictionary
                - symmetric mode so strategy and parameters are same for both agents
                - note that strategy superclass may have default parameter options - these are written to exp_data on conclusion of experiment
                
        '''
        
        agent_0 = agent.Agent(0, strategy_0, strategy_parameters_0)
        agent_1 = agent.Agent(1, strategy_1, strategy_parameters_1)

        
        '''
            - update exp_data summary with strategy options             [***]
                - can only do this after agents are instantiated, 
                    - cleaner to do it at the end than in the episode loop but need it set to get for filenames (written at each end of an episode) [TODO]
                    - ensures strategy superclass defaults are captured
            
        '''
    
        # if not exp_data["agent_parameters"]["agent_0"]["strategy"]:
            # exp_utility().set_exp_data_symmetric_agent_parameters(exp_data, [agent_0, agent_1])
            
            
            
        ''' 
            ---------------------------
            start single trial (1 episode)
            
                - update final timestep rewards in final_step() for both agent (in Trial.trial) 
                    This is normally done at start of each timestep in Trial.trial ( _trial.step() ), 
                    but after final timestep we do not enter _trial.step again in the timestep loop.
                
                - single_episode_outcomes_timestep_map can update by directly mapping into the outcome 
                - single_episode_outcomes_cumulative updates all four possible outcomes with running cumulative totals,
                    so calls a helper function to abstract four lines out of this section of code.
                
                heartbeat
                    - update heartbeat write (occurs every order_of_magnitude(timesteps) - 1 timesteps)
                    
                    
            Tournament Modifications
                - if we wish to preserve the timestep outcome record then the map is better than the cumulative as it will 
                compress far better due to the likelihood that there will be many long sequences and cycles of sequences.
                This does mean that post-processing of the map file will have to be converted to cumulative form if we wish to 
                directly access the cumulative view over the history of any episode.
                
                - it is better to convert cumulative to an in-memory running total rather than the full timestep sequence and 
                just write the total to file at the end of each tournament instance. So, we will write a file named after the 
                instance, and each line in that file will contain tournament_trial_id, episode_id, |CC|, |CD|, |DC|, |DD|
                
                - we do not by default write agent state or agent action_history for tournaments (grows too large, even compressed).
                
                - we do want total of agent reward_history at the very least. So again, maintain the running total fo each agent and 
                write to file at end of instance execution in similar format as cumulative file: write a file named after the 
                instance, and each line in that file will contain tournament_trial_id, episode_id, |a_0_reward|, |a_1_reward| 
                
                - we only write heartbeat if verbose is set to true. For most tournament executions this is not desired due to size 
                of heartbeat file unless the heartbeat interval is set to 0 (then only writes at heartbeat_start and heartbeat::ts==0)
                
                - The sum_episodes option allows the writing of total or full timestep sequence history for the agent reward_history and single_episode_outcomes_cumulative
                Default setting is TRUE, that is, to sum the episodes and write that to file in the format above.
                     
        '''
        
        _trial = trial.Trial((agent_0, agent_1), matrix)
        
        previous_step = []
        for t in range(timesteps):
            actions = _trial.step(t, previous_step)
            previous_step = actions
 
            if exp_data["exp_invocation"]["write_ts_map"]:
                single_episode_outcomes_timestep_map[exp_utility().map_actions_to_semantic_outcome(actions)][t] = 1
            
            # we don't want to maintain the entire history of cumulative timesteps.
            # just the current total for the given timestep, such that at the end of the episode we store the terminal sum for each outcome.
            #   one, we can recreate the cumulative history from the timestep_map file.
            #   two, we just want a quick total for each episode for later analysis, without having to go through each timestep_map file.
            #       if not sum_episodes, write whole cumulative sequence as per normal
            
            if not exp_data["exp_invocation"]["sum_episodes"]:
                exp_utility().append_timestep_outcome_to_single_episode_outcomes_cumulative(single_episode_outcomes_cumulative, _trial)
            
            
            if exp_data["exp_invocation"]['verbose_heartbeat']:
                exp_utility().heartbeat_tournament(exp_data, e, t, tournament_trial_id)
        
        _trial.final_step(t, previous_step)
        
        
        '''
            ^^ end single trial/episode
            ---------------------------
            then, 
            
            - update episode collections
                - per agent: total episode reward
                
            - write single episode collections
                - write agent action, reward history, strategy_state, agent_state
                - write single episode outcomes (cumulative and count)
                
            - which of these we write (and how) depends on runtime options.
                - for the super-trimmed down dataset the data structures are currently dictionaries whereas for the non-lean version they are arrays.
                - arrays can be written directly to csv file (and compressed, if that option is set), and code exists to unpackage them in run_obs/obs_util.py
                - extending the unpacking routines in run_obs/obs_util.py to handle .json as well is possible but:
                    a) the current code is working for csv, 
                    b) extending the current code to .json may or may not be a pain. (I don't really want to find out)
                    c) would have to convert from json data structure to csv for downstream run_obs processing anyway.
                - so, send the dictionary data structures to run_exp/exp_util.py routines, 
                    - insert a function _here_ (so, for tournaments only, if supertrim options are set) [TODO]
                    - this function converts from dict to csv, then sends to normal tournament write routines in exp_util.py [TODO]
                    
                - affects:
                    - 'sum_episodes'    -> exp_utility().write_agent_reward_history_sum_episodes_tournament(exp_data, agents_reward_episode_totals, tournament_trial_id)
                                        -> exp_utility().write_sum_episode_outcomes_cumulative_tournament(exp_data, e, sum_episode_outcomes_cumulative, tournament_trial_id)
                
                - does not affect:  (timestep_map is written as csv already)
                    - 'write_ts_map'    -> exp_utility().write_single_episode_outcomes_timestep_map_tournament(exp_data, e, single_episode_outcomes_timestep_map, tournament_trial_id)
                    
         '''
                
        if exp_data["exp_invocation"]["sum_episodes"]:
            for a in [agent_0, agent_1]:
                agents_reward_episode_totals[str(a.agent_id)][e] = sum(a.reward_history)
                
            # rather than append the timestep total for outcomes at every timestep, just get the episode total and insert to sum_episode_outcomes_cumulative
            exp_utility().append_episode_cumulative_outcome_sum(sum_episode_outcomes_cumulative, e, _trial)
        else:
            exp_utility().write_agent_reward_history_single_episode_tournament(exp_data, [agent_0, agent_1], e, tournament_trial_id)
            exp_utility().write_single_episode_outcomes_cumulative_tournament(exp_data, e, single_episode_outcomes_cumulative, tournament_trial_id)
        
        if exp_data["exp_invocation"]["write_ts_map"]:
            exp_utility().write_single_episode_outcomes_timestep_map_tournament(exp_data, e, single_episode_outcomes_timestep_map, tournament_trial_id)


    '''
        ^^ end all episodes
        ---------------------------
        then,
            see comments above re 'which we write'
        
    '''
    # write sum_episodes if applicable (if not, we have already written single episode files at end of each episode)
    if exp_data["exp_invocation"]["sum_episodes"]:
        for a in [agent_0, agent_1]:
            exp_utility().write_agent_reward_history_sum_episodes_tournament(exp_data, a.agent_id, agents_reward_episode_totals[str(a.agent_id)], tournament_trial_id)
            
        for o in sum_episode_outcomes_cumulative.keys():
            exp_utility().write_sum_episode_outcomes_cumulative_tournament(exp_data, o, sum_episode_outcomes_cumulative[o], tournament_trial_id)
        
    if exp_data["exp_invocation"]['verbose_heartbeat']:
        exp_utility().heartbeat_end_tournament(exp_data, tournament_trial_id) 
   
   

def parse_input(exp_data, argv):

    try:
        options, args = getopt.getopt(argv, "hj:w:f:g:r:e:t:p:k:z:c:l:m:s:b:v:d:i:", ["pbs_jobstr", "pbs_scratch_dir", "family", "gameform", "reward_type", "episodes", "timesteps", "strategy_set_input", "write_state", "compress_writes", "write_2_heartbeats", "localhost", "mode", "sum_episodes", "write_ts_map", "verbose", "verbose_heartbeat", "heartbeat_interval"])
        print(os.path.basename(__file__) + ":: args", options)
    except getopt.GetoptError:
        print(os.path.basename(__file__) + ":: error in input, try -h for help")
        sys.exit(2)
        
    for opt, arg in options:
    
        if opt == '-h':
            print("usage: " + os.path.basename(__file__) + " \r\n \
            -j <pbs_jobstr [__JOBSTR__]> | \r\n \
            -w <pbs_scratch_dir [__PATH__]> | \r\n \
            -f <family [axelrod, crandall, bandit, rlmethods, all]>] \r\n \
            -g <gameform [pd|staghunt|...|chicken|g{nnn}|all_topology ]> \r\n \
            -r <reward_type [scalar|ordinal|ordinal_transform ]> \r\n \
            -e <episodes> \r\n \
            -t <timesteps> \r\n \
            -p <strategy_set_input ['{file}.strategy_set'|'na']> \r\n \
            -k <write_state> boolean (default is false)\r\n \
            -z <compress_writes> boolean (default is true) \r\n \
            -c <write_2_heartbeats> boolean (default is true) \r\n \
            -l <localhost> boolean (default is false) \r\n \
            -m <mode> [symmetric_selfplay | asymmetric_selfplay] \r\n \
            -s <sum_episodes> \r\n \
            -b <write_ts_map> \r\n \
            -v <verbose> \r\n \
            -d <verbose_heartbeat> \r\n \
            -i <heartbeat_interval> \r\n \
            - minimal: (-s) {strategy} -g {gameform} -e {n} -t {n} -p {file_name} \r\n \
            - required: -m {options} -s {strategy} -g {gameform} -e {n} -t {n} -p {} \r\n \
        ")
        
        elif opt in ('-j', '--pbs_jobstr'):
            exp_data["job_parameters"]["pbs_jobstr"] = arg
            
        elif opt in ('-w', '--pbs_scratch_dir'):
            exp_data["job_parameters"]["pbs_scratch_dir"] = arg
            
        elif opt in ('-f', '--family'):
            exp_data["exp_invocation"]["family"] = arg
            
        elif opt in ('-g', '--gameform'):
            exp_data["exp_invocation"]["gameform"] = arg
            
        elif opt in ('-r', '--reward_type'):
            exp_data["exp_invocation"]["reward_type"] = arg
            
        elif opt in ('-e', '--episodes'):
            exp_data["exp_invocation"]["episodes"] = int(arg)
            
        elif opt in ('-t', '--timesteps'):
            exp_data["exp_invocation"]["timesteps"] = int(arg)
            
        elif opt in ('-p', '--strategy_set_input'):
            exp_data["exp_invocation"]["strategy_set_input"] = arg
        
        elif opt in ('-k', '--write_state'):
            if arg == 'true':
                exp_data["exp_invocation"]["write_state"] = True
            
        elif opt in ('-z', '--compress_writes'):
            if arg == 'false':
                exp_data["exp_invocation"]["compress_writes"] = False

        elif opt in ('-c', '--write_2_heartbeats'):
            if arg == 'false':
                exp_data["exp_invocation"]["write_2_heartbeats"] = False 

        elif opt in ('-l', '--localhost'):
            if arg == 'true':
                exp_data["exp_invocation"]["localhost"] = True                 
        
        elif opt in ('-m', '--mode'):
            exp_data["exp_invocation"]["mode"] = arg    
        
        elif opt in ('-s', '--sum_episodes'):
            if arg == 'false':
                exp_data["exp_invocation"]["sum_episodes"] = False                
        
        elif opt in ('-b', '--write_ts_map'):
            if arg == 'false':
                exp_data["exp_invocation"]["write_ts_map"] = False
                
        elif opt in ('-v', '--verbose'):
            if arg == 'true':
                exp_data["exp_invocation"]["verbose"] = True
                
        elif opt in ('-d', '--verbose_heartbeat'):
            if arg == 'true':
                exp_data["exp_invocation"]["verbose_heartbeat"] = True                
                
        elif opt in ('-i', '--heartbeat_interval'):
            exp_data["exp_invocation"]["heartbeat_interval"] = int(arg)
    
    if not options:
        print(os.path.basename(__file__) + ":: error in input: no options supplied, try -h for help")
    else:
        exp_data["exp_invocation"]["exp_args"] = options
        
    if exp_data["job_parameters"]["pbs_jobstr"] == "":
        print(os.path.basename(__file__) + ":: error in input: pbs_jobstr is required, use -j __STR__ or try -h for help")
        sys.exit(2)
        
    if exp_data["job_parameters"]["pbs_scratch_dir"] == "" and not exp_data["exp_invocation"]["localhost"]:
        print(os.path.basename(__file__) + ":: error in input: pbs_scratch_dir is required, use -w __STR__ or try -h for help")
        sys.exit(2) 
        
    if exp_data["exp_invocation"]["gameform"] == "":
        print(os.path.basename(__file__) + ":: error in input: gameform is required, use -g __STR__ or try -h for help")
        sys.exit(2)
        
    if exp_data["exp_invocation"]["reward_type"] == "":
        print(os.path.basename(__file__) + ":: error in input: reward_type is required, use -r __STR__ or try -h for help")
        sys.exit(2)
        
    if exp_data["exp_invocation"]["episodes"] == 0 or exp_data["exp_invocation"]["episodes"] == '':
        print(os.path.basename(__file__) + ":: error in input - requires episodes, use -e __INT__ or try -h for help")
        sys.exit(2)
        
    if exp_data["exp_invocation"]["timesteps"] == 0 or exp_data["exp_invocation"]["timesteps"] == '':
        print(os.path.basename(__file__) + ":: error in input - requires timesteps, use -t __INT__ or try -h for help")
        sys.exit(2)
        
    if exp_data["exp_invocation"]["strategy_set_input"] == "" :
        print(os.path.basename(__file__) + ":: error in input: strategy_set_input filename is required, use -p __STR__ or try -h for help")
        sys.exit(2)
    
    
def initialise_exp_data():    
    
    exp_time_start_ns = time_ns()
    
    return {
        "exp_id"                    : "",
        "journal_output_filename"   : "",
        "job_parameters"            : {
            "pbs_jobstr"            : "",
            "pbs_jobid"             : "",
            "pbs_parent_jobid"      : "",
            "pbs_sub_jobid"         : 0,
            "pbs_scratch_dir"       : "",
            "hpc_name"              : "",
            "data_filename_prefix"  : "",
            "job_dir"               : "",
            "job_args"              : "",
            "heartbeat_path"        : "",
            "journal_path"          : "",
            "heartbeat_interval"    : 0
        },
        "exp_invocation"          : {
            "filename"              : __file__,
            "exp_args"              : "",
            "exp_time_start_hr"     : datetime.fromtimestamp(exp_time_start_ns / 1E9).strftime("%d%m%Y-%H%M%S"),
            "exp_time_end_hr"       : "",
            "exp_time_start_ns"     : str(exp_time_start_ns),
            "exp_time_end_ns"       : "",
            "process_start_ns"      : process_time_ns(),
            "process_end_ns"        : 0,
            "exp_type"              : re.search(r"exp_([A-Za-z_\s]*)", os.path.basename(__file__))[1],
            "family"                : "",
            "gameform"              : "",
            "reward_type"           : "",
            "episodes"              : 0,
            "timesteps"             : 0,
            "strategy_set_input"    : "",
            "strategy_list"         : [],
            "write_state"           : False,
            "compress_writes"       : True,
            "write_2_heartbeats"    : True,
            "localhost"             : False,
            "home"				    : "",
            "basepath"				: "",
            "mode"                  : "",
            "sum_episodes"          : True,
            "write_ts_map"          : True,
            "verbose"               : False,
            "verbose_heartbeat"     : False,
            "heartbeat_interval"    : 0
        },
        "exp_tournament_trials"  : {
                "0"                     : {
                    "exp_tournament_trial_id"          : 0,
                    "agents"          : {
                        "agent_0"             : { 
                            "strategy"          : "",
                            "strategy_parameters" : "",
                        },
                        "agent_1"             : {
                            "strategy"          : "",
                            "strategy_parameters" : "",
                        },
                    },
                },
            },

    }

    
if __name__ == '__main__':
    main(sys.argv[1:]) 