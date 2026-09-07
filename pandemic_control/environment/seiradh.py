import os
import numpy as np

from scipy.integrate import odeint
from typing import (
    Any,
    Dict, 
    Tuple
)

from .base import Base_Env


class SEIRADH_Env(Base_Env):
   
    def __init__(self, cfg: Dict | os.PathLike) -> None:
        super().__init__(cfg)
        
        #  SEIRADH parameters
        if not hasattr(self, f"I_a0"):
            self.I_a0 = 0
        if not hasattr(self, f"I_s0"):
            self.I_s0 = 1
        if not hasattr(self, f"R0"):
            self.R0 = 0
        if not hasattr(self, f"E0"):
            self.E0 = 0
        if not hasattr(self, f"D0"):
            self.D0 = 0
        if not hasattr(self, f"H0"):
            self.H0 = 0
        if not hasattr(self, f"S0"):
            self.S0 = self.N - sum([getattr(self, f"{k}0", 0) for k in ['E','I_a','I_s','R','D','H']])
        for comp in  ['S', 'E','I_a','I_s','R','D', 'H']:
            if not hasattr(self, f"{comp}"):
                setattr(self, f"{comp}", getattr(self, f"{comp}0"))
        # beta : infection rate
        # omega : vaccination rate
        # rho : vaccine inefficacity
        # gamma : recovery rate for symptomatic, asymptomatic and hospitalized. 
        # delta : incubation rate
        # theta : hospitalization rate
        # mu : death rate for symptomatics and hospitalized
        # sigma : losing immunity rate = 1/number of days before losing natural immunity

        # Default values
        if not hasattr(self, f"beta"):
            self.beta = 0.8
        if (not hasattr(self, f"gamma")) or (type(self.gamma) != type([])) or (len(self.gamma) != 3):
            self.gamma = [1./10, 1./15]
        if not hasattr(self, f"delta"):
            self.delta = 1/5.1
        if not hasattr(self, f"theta"):
            self.theta = 1/5.9
        if (not hasattr(self, f"mu"))  or (type(self.mu) != type([])) or (len(self.mu) != 2):
            self.mu = [1/10, 1/14]
        if not hasattr(self, f"sigma"):
            self.sigma = 1/180
        if not hasattr(self, f"rho"):
            self.sigma = 0.1

        
        # Probabilities.
        # Configuration files spell this key either "probas" or "probs"; both are
        # accepted so that the values in the configuration are always the ones the
        # model integrates.
        if not hasattr(self, "probas"):
            self.probas = getattr(self, "probs", [
                0.8,    # Probability of showing symptoms
                0.3,    # Probability of hospitalization
                0.02,   # Probability of death for symptomatic hospitalized
                0.3,    # Death probability after hospitalization
            ])
        self.probs = self.probas
        
        # History
        self.list_S = []
        self.list_E = []
        self.list_I_a = []
        self.list_I_s = []
        self.list_H = []
        self.list_R = []
        self.list_D = []
        

    def step(self, action: float) -> Tuple[
        np.ndarray, 
        float, 
        bool, 
        bool, 
        dict[str, Any]
        ]:
        
        done = False
        #  Make an action
        self.choose_action(action)
        #  Update observation
        y = self.S, self.E, self.I_a, self.I_s, self.H, self.R, self.D #on sauvegarde old observation        
        ret = odeint(self.deriv, y, self.time) 

        self.S, self.E, self.I_a, self.I_s, self.H, self.R, self.D = ret[-1]
        
        observation = np.array([self.S,
                                self.E, 
                                self.I_a, 
                                self.I_s,
                                self.H, 
                                self.R, 
                                self.D], dtype=np.float32)
        
        if round(self.S + self.E + self.I_a + self.I_s + self.H + self.R + self.D) != self.N:
            raise Exception("The sum of compartiments isn't equal to N")
            
        #  Calculate reward
        rew = self.reward(action)
        
        self.steps += 1
        if self.steps == self.max_steps:
            done = True
        
        self.update_history(ret, rew, action)
        return observation, rew, done, False, {}
    
    def choose_action(self, choice) -> None:
        self.beta = self.actions[int(choice)][0]

    
    def reward(self, action: float) -> float:
        #  The economic reward : the agent is penalized for a high restriction level
        #  The health cost : the agent is penalized for a high death/infection toll
        iw = self.health_weights[0]
        hw = self.health_weights[1]
        dw = self.health_weights[2]
        
        hosp_margin = 0.7*self.hosp_cap
        eco_cost = self.actions[int(action)][1]
        inf_cost = -(self.I_a+self.I_s)/self.N
        death_cost = -self.D/self.N
        self.economic_cost += self.days*[eco_cost]
        if (self.H < (0.7 * self.hosp_cap)):
            #hospital_cost = - self.H/self.N
            hospital_cost = 0
        else :
            hospital_cost = -(self.H - hosp_margin)/hosp_margin
        
        if (self.N*0.10<(self.I_a+self.I_s)): 
            eco_cost = eco_cost/1.8 #donner encore moins d'importance à l'économie
        else:
            inf_cost = 0
            
        health_cost = iw*inf_cost + dw*death_cost + hw*hospital_cost
        
        self.infected_cost += self.days*[inf_cost]
        self.death_cost += self.days*[death_cost]
        self.hospt_cost += self.days*[hospital_cost]
        self.health_cost += self.days*[health_cost]
        
        
        return self.trade_off_weights[0] * health_cost + self.trade_off_weights[1] * eco_cost 
        
        
        
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed)

        self.steps = 0
        self.beta = 0.8
        self.S, self.E, self.I_a, self.I_s, self.H, self.R, self.D = self.S0, self.E0, self.I_a0, self.I_s0, self.H0, self.R0, self.D0
        
        #inits for plotting
        self.list_S = []
        self.list_E = []
        self.list_I_a = []
        self.list_I_s = []
        self.list_H = []
        self.list_R = []
        self.list_D = []
        
        self.economic_cost =[]
        self.health_cost = []
        self.infected_cost = []
        self.death_cost = []
        self.hospt_cost = []
        self.rewards = []
        self.list_actions = []
        self.list_betas = []

        observation = np.array([self.S,
                                self.E, 
                                self.I_a, 
                                self.I_s, 
                                self.H, 
                                self.R, 
                                self.D], dtype=np.float32)
   
        return observation, {} 
    

    def build_env_data(self) -> Dict[str, Any]:
        model_data = {
            'Environment': [self.env_name] * len(self.list_S),
            'N': [self.N]  * len(self.list_S),
            'Hosp_Cap': [self.hosp_cap] * len(self.list_S),
            'Susceptible': self.list_S,
            'Exposed' : self.list_E,
            'Asymptomatic': self.list_I_a,
            'Symptomatic': self.list_I_s,
            'Recovered': self.list_R,
            'Hospitalized': self.list_H,
            'Deceased': self.list_D,
            'Infected': np.add(self.list_I_a, self.list_I_s),
            'Days': np.array(range(1, self.days*self.steps+1)),
            'Economy': self.economic_cost,
            'Health': self.health_cost,
            'Reward' : self.rewards,
            'Infected_rew': self.infected_cost,
            'Hospitalized_rew': self.hospt_cost,
            'Deceased_rew': self.death_cost,
            'Actions' : self.list_actions
            }
        
        return model_data
    
    def update_history(self, ret:np.ndarray, rew: float, action: float) -> None:
        self.list_S = [*self.list_S,*ret[1:].T[0]]
        self.list_E = [*self.list_E,*ret[1:].T[1]]
        self.list_I_a = [*self.list_I_a,*ret[1:].T[2]]
        self.list_I_s = [*self.list_I_s,*ret[1:].T[3]]
        self.list_H = [*self.list_H,*ret[1:].T[4]]
        self.list_R = [*self.list_R,*ret[1:].T[5]]
        self.list_D = [*self.list_D,*ret[1:].T[6]]
        self.list_actions += self.days*[int(action)]
        self.list_betas += self.days*[self.beta]
        self.rewards += self.days*[rew]

    def deriv(self, y: Tuple[float], t: int) -> Tuple[
        float, 
        float,
        float,
        float,
        float,
        float
        ]:
        S, E, I_a, I_s, H, R , D = y
        p_s, p_h, p_d, p_dh =  self.probas
        N, beta, gamma, delta, theta, mu, sigma =  self.N, self.beta, self.gamma, self.delta, self.theta, self.mu, self.sigma
        
        dSdt = sigma * R - beta * S *(I_a + I_s)/N
        dEdt = beta * S * (I_a + I_s)/N - delta*E
        dI_adt = (1-p_s)*delta * E - gamma[0] * I_a
        dI_sdt = p_s*delta * E - ((1-(p_d+p_h))*gamma[1] + p_d*mu[0] + p_h*theta ) * I_s
        dHdt = p_h*theta * I_s - ( p_dh*mu[1] + (1-p_dh)*gamma[2]) * H
        dRdt = gamma[0] * I_a + (1-(p_d+p_h))*gamma[1] * I_s + (1-p_dh)*gamma[2] * H - sigma * R
        dDdt = p_d*mu[0] * I_s + p_dh*mu[1] * H
        
        return dSdt, dEdt, dI_adt, dI_sdt, dHdt, dRdt , dDdt
