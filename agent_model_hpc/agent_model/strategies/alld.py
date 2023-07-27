#!/usr/bin/env python3
# File: alld.py

from .strategy import Strategy

class Alld(Strategy):
        
    def __init__(self, strategy, strategy_options):
        super().__init__(strategy, strategy_options)
    
    def action(self, agent, previous_step):
        super().action(agent, previous_step)
        return 1
        
if __name__ == '__main__':
    print(self.options["name"])