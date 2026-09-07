import os
import numpy as np

from scipy.integrate import odeint
from typing import (
    Any,
    Dict, 
    Tuple
)

from .base import Base_Env


# Main model
class SEIRADHV_Env(Base_Env):
    def __init__(self, cfg: Dict | os.PathLike) -> None:
        super().__init__(cfg)

        self.n_resets = 0
        
        #  SEIRADHV parameters
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
        if not hasattr(self, f"V0"):
            self.V0 = 0
        if not hasattr(self, f"S0"):
            self.S0 = self.N - sum([getattr(self, f"{k}0", 0) for k in ['E','I_a','I_s','R','D','H','V']])
        for comp in ['S','E','I_a','I_s','R','D','H','V']:
            if not hasattr(self, f"{comp}"):
                setattr(self, f"{comp}", getattr(self, f"{comp}0"))
        
        # TODO: Figure out what are those variables for
        self.Hc = self.H0
        self.Ic = self.I_a0 + self.I_s0

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
        self.list_V = []
        self.list_E = []
        self.list_I_a = []
        self.list_I_s = []
        self.list_H = []
        self.list_R = []
        self.list_D = []

        # beta : infection rate
        # omega : vaccination rate
        # rho : vaccine inefficacity
        # gamma : recovery rate pour les asymptomatiques, symptomatiques et hospitalisés
        # delta : incubation rate
        # theta : hospitalization rate
        # mu : death rate pour les symptomatiques et hospitalisés
        # sigma : losing immunity rate = 1/nombre de jours avant de perdre l'immunité naturelle

        if not hasattr(self, f"beta"):
            self.beta = 0.8
            self.beta_0 = self.beta
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


        # Extra fields
        self.list_I_cumul = []
        self.list_H_cumul = []
        self.list_R_cumul = []
        self.list_V_cumul = []
        self.list_D_cumul = []
        self.infection_cost = []
        self.death_cost = []
        self.hospt_cost = []

        self.I_cumul = 0
        self.H_cumul = 0
        self.economic_contribution = self.N
        self.economic_cost_cumul = 0
        self.prev_action = 0
        

    
    """ Method used to update initial conditions after the environment is initialized """
    def update_initial_conditions(
        self, 
        V0: int, 
        E0: int, 
        I_a0: int, 
        I_s0: int, 
        H0: int, 
        R0: int, 
        D0: int,
        ) -> None:
        s = V0 + E0 + I_a0 + I_s0 + H0 + R0 + D0
        if V0 < 0 or E0 < 0 or I_a0 < 0 or I_s0 < 0 or H0 < 0 or R0 < 0 or D0 < 0:
            raise ValueError(f"Error: the initial variables should all be positive")
        elif (V0 + E0 + I_a0 + I_s0 + H0 + R0 + D0 > self.N):
            raise ValueError(f"Error: the sum of initial variables '{s}' should be equal to the population '{self.N}'")
        else:
            self.V0 = V0
            self.E0 = E0
            self.I_a0 = I_a0
            self.I_s0 = I_s0
            self.H0 = H0
            self.R0 = R0
            self.D0 = D0
            self.S0 = self.N - s
        


    def update_history(self, ret:np.ndarray, rew: float, action: float) -> None:
        self.list_S = [*self.list_S, *ret[1:].T[0]]
        self.list_V = [*self.list_V, *ret[1:].T[1]]
        self.list_E = [*self.list_E, *ret[1:].T[2]]
        self.list_I_a = [*self.list_I_a, *ret[1:].T[3]]
        self.list_I_s = [*self.list_I_s, *ret[1:].T[4]]
        self.list_H = [*self.list_H, *ret[1:].T[5]]
        self.list_R = [*self.list_R, *ret[1:].T[6]]
        self.list_D = [*self.list_D, *ret[1:].T[7]]

        self.list_I_cumul = [*self.list_I_cumul, *ret[1:].T[8]]
        self.list_H_cumul = [*self.list_H_cumul, *ret[1:].T[5]]
        self.list_R_cumul = [*self.list_R_cumul, *ret[1:].T[6]]
        self.list_V_cumul = [*self.list_V_cumul, *ret[1:].T[1]]
        self.list_D_cumul = [*self.list_D_cumul, *ret[1:].T[7]]
        

        self.rewards += self.days*[rew]
        self.list_actions += self.days*[int(action)]
        self.list_betas += self.days*[self.beta]



    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

        done = False

        self.choose_action(action)

        # Save previous state of the model
        y = self.S, self.V, self.E, self.I_a, self.I_s, self.H, self.R, self.D, self.Ic, self.Hc
        prev_obs = y

        #  Update observation
        ret = odeint(self.deriv, y, self.time)

        #expliquer derive

        self.S, self.V, self.E, self.I_a, self.I_s, self.H, self.R, self.D, self.Ic, self.Hc = ret[-1]
        
        ## Adding this line: to remove if this leads to errors!
        #self.S, self.V, self.E, self.I_a, self.I_s, self.H, self.R, self.D, self.Ic, self.Hc = round(self.S), round(self.V), round(self.E), round(self.I_a), round(self.I_s), round(self.H), round(self.R), round(self.D), round(self.Ic), round(self.Hc)
        observation = np.array([self.S,
                                self.V,
                                self.E,
                                self.I_a,
                                self.I_s,
                                self.H,
                                self.R,
                                self.D], dtype=np.float32)
        if np.round(observation.sum()) !=  self.N:
            raise Exception(f"The sum of compartments ({round(observation.sum()) }) isn't equal to N ({self.N})")

        rew = self.reward(action, prev_obs)
        self.prev_action = action
        

        self.steps += 1
        if (self.steps % self.max_steps == 0): 
            done = True
    
        self.update_history(ret, rew, action)
        return observation, rew, done, False, {}  # obs, reward, terminated, truncated, info

    def choose_action(self, choice: float):
        self.beta = self.actions[int(choice)][0]


    def reward(self, action: float, prev_obs: np.ndarray):
        iw = self.health_weights[0]
        hw = self.health_weights[1]
        dw = self.health_weights[2]
        hosp_margin = 0.7*self.hosp_cap
        eco_cost = self.actions[int(action)][1]
        inf_cost = -(self.I_a+self.I_s)/self.N
        death_cost = -self.D/self.N

        if (self.H < (0.7 * self.hosp_cap)):
            #hospital_cost = - self.H/self.N
            hospital_cost = 0
        else :
            hospital_cost = -(self.H - hosp_margin)/hosp_margin
        
        prc = 0.1
        if (self.N*prc>(self.I_a+self.I_s)): 
            inf_cost = 0
                
        health_cost = iw*inf_cost + dw*death_cost + hw*hospital_cost

        self.infection_cost += self.days*[inf_cost]
        self.death_cost += self.days*[death_cost]
        self.hospt_cost += self.days*[hospital_cost]
        self.health_cost += self.days*[health_cost]
        self.economic_cost += self.days*[eco_cost]

        self.economic_cost_cumul +=eco_cost

        return self.trade_off_weights[0] * health_cost + self.trade_off_weights[1] * eco_cost 

    def reset(self, seed:int | None = None):
        super().reset(seed)

        self.steps = 0

        self.I_cumul = 0
        self.H_cumul = 0

        self.S, self.V, self.E, self.I_a, self.I_s, self.H, self.R, self.D, self.Ic, self.Hc = self.S0, self.V0, self.E0, self.I_a0, self.I_s0, self.H0, self.R0, self.D0, self.I_a0, self.H0
        self.beta = self.beta_0
        self.prev_action = 0

        # inits for plotting
        self.list_S = []
        self.list_V = []
        self.list_E = []
        self.list_I_a = []
        self.list_I_s = []
        self.list_H = []
        self.list_R = []
        self.list_D = []

        self.list_I_cumul = []
        self.list_H_cumul = []
        self.list_R_cumul = []
        self.list_V_cumul = []
        self.list_D_cumul = []


        self.rewards = []
        self.economic_cost = []
        self.health_cost = []

        self.economic_contribution = self.N
        self.economic_cost_cumul = 0
        
        self.infection_cost = []
        self.death_cost = []
        self.hospt_cost = []

        self.list_actions = []
        self.list_betas = []

        observation = np.array(
            [self.S, self.V, self.E, self.I_a, self.I_s, self.H, self.R, self.D], dtype=np.float32)
        
        self.n_resets += 1
        
        return observation, {}  # obersvation:PlaceHolder("ObsType"), info: Dict
    
    def build_env_data(self):
        model_data = {
            'Environment': [self.env_name] * len(self.list_S),
            'N': [self.N] * len(self.list_S),
            'Hosp_Cap': [self.hosp_cap] * len(self.list_S),
            'Susceptible': self.list_S,
            'Vaccinated': self.list_V,
            'Exposed': self.list_E,
            'Asymptomatic': self.list_I_a,
            'Symptomatic': self.list_I_s,
            'Hospitalized': self.list_H,
            'Recovered': self.list_R,
            'Deceased': self.list_D,
            'Infected': np.add(self.list_I_a, self.list_I_s),
            'Infected_cumul' : np.cumsum(np.add(self.list_I_a, self.list_I_s)),
            'Deceased_cumul' : np.cumsum(self.list_D),
            'Hospitalized_cumul': np.cumsum(self.list_H),
            'Recovered_cumul' : np.cumsum(self.list_R),
            'Vaccinated_cumul' : np.cumsum(self.list_V),
            'Days': np.array(range(1, (self.days*self.steps)+1)),
            'Reward': self.rewards,
            'Actions': self.list_actions,
            'Economy': self.economic_cost,
            'Health': self.health_cost,
            'Infected_rew': self.infection_cost,
            'Hospitalized_rew': self.hospt_cost,
            'Deceased_rew': self.death_cost,
            'Health_cumul': np.cumsum(self.health_cost),
            'Economy_cumul': np.cumsum(self.economic_cost),
            'Reward_cumul': np.cumsum(self.rewards),
            }
        return model_data

    def deriv(self, y: Tuple[float], t: int):
        S, V, E, I_a, I_s, H, R, D, Ic, Hc = y
        p_s, p_h, p_d, p_dh = self.probas
        N, beta, gamma, omega, rho, delta, theta, mu, sigma = (
            self.N, 
            self.beta, 
            self.gamma, 
            self.omega, 
            self.rho, 
            self.delta, 
            self.theta, 
            self.mu, 
            self.sigma
        )

        # Differential equations
        dSdt = sigma * R - beta * S * (I_a + I_s)/N - omega*S
        dVdt = omega*S - rho*V*(I_a + I_s)/N
        dEdt = beta * S * (I_a + I_s)/N - delta*E + rho*V*(I_a+I_s)/N
        dI_adt = (1-p_s)*delta * E - gamma[0] * I_a
        dI_sdt = p_s*delta * E - \
            ((1-(p_d+p_h))*gamma[1] + p_d*mu[0] + p_h*theta) * I_s

        dHdt = p_h*theta * I_s - (p_dh*mu[1] + (1-p_dh)*gamma[2]) * H


        dRdt = gamma[0] * I_a + (1-(p_d+p_h))*gamma[1] * \
            I_s + (1-p_dh)*gamma[2] * H - sigma * R

        dDdt = p_d*mu[0] * I_s + p_dh*mu[1] * H

        
        dIcdt = delta * E
        dHcdt = p_h*theta * I_s
        

        return dSdt, dVdt, dEdt, dI_adt, dI_sdt, dHdt, dRdt, dDdt, dIcdt, dHcdt
