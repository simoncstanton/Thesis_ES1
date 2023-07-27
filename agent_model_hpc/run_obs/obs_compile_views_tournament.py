#!/usr/bin/env python3
'''
    obs_compile_views_tournament

    orderset - a collection of first statistics and reports on an experiment
    
    runs all operations in each
        
'''
import sys

from run_obs.view_components import obs_generate_view_tournament_outcomes as tourney_outcomes
from run_obs.view_components import obs_generate_view_tournament_dataset_game_theoretic as game_theoretic
from run_obs.view_components import obs_generate_view_tournament_dataset_bandits as bandits
from run_obs.view_components import obs_generate_view_tournament_dataset_rl_methods as rlmethods
from run_obs.view_components import obs_generate_view_tournament_dataset_tourneyfour as tourneyfour
#from run_obs.view_components import obs_generate_view_ts_r as ts_rewards



def main(argv):
    
    
    tourney_outcomes.main(argv[1:])
    #game_theoretic.main(argv[1:])
    tourneyfour.main(argv[1:])
    #bandits.main(argv[1:])
    
   







if __name__ == '__main__':
    main(sys.argv)  