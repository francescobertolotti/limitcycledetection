import numpy as np

class Agent:
    def __init__(self, init_cooperate, memory, mean_init_payoffs):
        self.strategy = 'cooperate' if np.random.random() > init_cooperate else 'defect'
        self.payoff = [mean_init_payoffs for r in range(memory)]

    def play(self, other_agent, payoffs):
        po_cc, po_cd, po_dc, po_dd = payoffs
        if self.strategy == "cooperate":
            if other_agent.strategy == "cooperate":
                self.payoff.append(po_cc)
                other_agent.payoff.append(po_cc)
            else:
                self.payoff.append(po_cd)
                other_agent.payoff.append(po_dc)
        else:
            if other_agent.strategy == "cooperate":
                self.payoff.append(po_dc)
                other_agent.payoff.append(po_cd)
            else:
                self.payoff.append(po_dd)
                other_agent.payoff.append(po_dd)

class Model:
    def __init__(self, n_agents, n_rounds, init_cooperate, memory, po_cc, po_cd, po_dc, po_dd):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.memory = memory
        self.payoffs = [po_cc, po_cd, po_dc, po_dd]
        self.agents = [Agent(init_cooperate, memory, np.mean(self.payoffs)) for _ in range(n_agents)]
        

        #globals
        self.mean_payoffs = []
        self.pct_cooperate = []
    
    def play_round(self):
        
        np.random.shuffle(self.agents)
        self.agents[0].play(self.agents[1], self.payoffs)

        mean_payoffs = sum([sum(a.payoff) for a in self.agents])/(self.n_agents * self.memory)
        playing_agents = [self.agents[0], self.agents[1]]

        for agent in playing_agents:
            if np.mean(agent.payoff) < mean_payoffs:
                agent.strategy = np.random.choice(["cooperate", "defect"])
            agent.payoff = agent.payoff[1:] #forgot the first one
            
    def run(self):
        for _ in range(self.n_rounds):
            self.play_round()
            self.mean_payoffs.append(np.mean([agent.payoff for agent in self.agents]))
            self.pct_cooperate.append(len([agent for agent in self.agents if agent.strategy == 'cooperate']) / self.n_agents)


    def output(self):
        return self.mean_payoffs, self.pct_cooperate 