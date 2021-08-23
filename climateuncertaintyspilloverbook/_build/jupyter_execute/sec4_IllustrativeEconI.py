#!/usr/bin/env python
# coding: utf-8

# The notebooks proceed with three example economies with different features:
# 
# - the first one features damage function uncertainty and its resolution (demostrated in this notebook);
# - the second features a novel uncertainty decomposition that incorporates robustness to model ambiguity and misspecification ([Section 5](sec5_IllustrativeEconII.ipynb));
# - the investigates the impact of uncertain advances in the availability of less carbon-intensive technologies ([Section 6](sec6_IllustrativeEconIII.ipynb)).
# 
# # 4 Illustrative economy I: uncertain damages
# 
# 
# We pose  an  $AK$ technology for which output is
# proportional to capital and can be allocated  between investment  and consumption. Capital in this specification should be broadly conceived. Suppose that there are adjustment costs to capital that are represented as the product of capital
# times a quadratic function of the investment-capital ratio.
# Given the output constraint and capital evolution imposed by  the $AK$ technology, it suffices to let  the planner choose the
# investment-capital ratio.
# 
# 
# Formally, "undamaged" capital evolves as
# 
# $$
# d K_t =  K_t   \left[ \mu_k (Z_t) dt + \left({\frac {I_t}{K_t}} \right)dt - {\frac { \kappa} 2} \left( {\frac {I_t} {K_t}} \right)^2 dt
# + \sigma_k(Z_t) dW_t^k \right]
# $$
# 
# where $K_t$ is the capital stock and $I_t$ is investment.
# The capital evolution expressed in logarithms is
# 
# $$
# d\log K_t =  \left[ \mu_k (Z_t)    + \left({\frac {I_t}{K_t}} \right)  -
# {\frac { \kappa} 2} \left( {\frac {I_t} {K_t}} \right)^2 \right] dt -  {\frac  {\vert \sigma_k(Z_t) \vert^2}  2}dt+ \sigma_k(Z_t) dW_t^k ,
# $$
# 
# The sum of consumption, $C_t$, and investment, $I_t$, are constrained to be proportional to capital:
# 
# $$
# C_t + I_t = \alpha K_t
# $$
# 
# 
# Next, we consider environmental damages.
# We suppose that temperature shifts proportionately consumption and capital by a multiplicative factor
# $N_t$  that captures damages to the productive capacity induced by climate change.  For instance, the
# damage adjusted consumption is ${\widetilde C}_t =  {\frac {C_t}{N_t}}$ and the damage adjusted capital is ${\widetilde K}_t = {\frac {{K}_t }{N_t}}$.  
# 
# 
# Planner preferences are time-separable with  a unitary elasticity of substitution. The planner's instantaneous utility from "damaged consumption" and emissions is given by:
# 
# $$
# \begin{align*}
# &  (1-\eta) \log {\tilde C}_t +  \eta \log {\mathcal E}_t   \cr & = (1-\eta)( \log C_t -\log K_t ) +  (1-\eta)( \log K_t - \log N_t)   + \eta \log {\mathcal E}_t
# \end{align*}
# $$
# 
# We let $\delta$ be the subjective rate of discount used in preferences.
# We can think of emissions and consumption as distinct goods, or we can think of $\widetilde{C}_t$ as an intermediate good that when combined with emissions determines final consumption.
# 
# 
# >**Note**
# > 
# >*We obtain a further simplication by letting:*
# >
# >$$\widetilde{\mathcal{E}}_t = \mathcal{E}_t (\iota_y \cdot Z_t)$$
# >
# >*We use $\widetilde{\mathcal{E}}_t$ as the control variable and then deduce the implications for $\mathcal{E}_t$*.
# 
# 
# ## 4.1 HJB equations and robustness
# 
# The uncertainty that we consider has a single jump point after which the damage function uncertainty is revealed.  This leads us to compute continuation value functions conditioned on each of the damage function specifications.  These continuation value functions then are used to summarize post-jump outcomes when we compute the initial value function.  We describe the Hamilton-Jacobi-Bellman (HJB) equations for each of these steps in what follows. The computational methods are described in the [appendix](appendices.ipynb).
# 
# 
# The parameter values are as follows:
# 
# | Parameters | values |
# | :---:| :---|
# |$\delta$ |  0.01 |
# |$\eta$ | 0.032 | 
# |$\varsigma'$| [2.23, 0, 0]|
# 
# Damage parameters are described in section 3 (TODO).
# 
# The penalty paramters are $\xi_a$ and $\xi_r$. Without specifically pointed out, $\xi_a = 0.01$ in this example. And the $\xi_r$ values we experiment with are $\{+ \infty, 5, 1, 0.3\}$.
# 
# ### 4.1.1 Post-jump continuation value functions
# 
# 
# Conditioned on each of the damage functions, $m = 1, 2, \dots, 20$. Solve for the corresponding $\phi_m(y)$:
# 
# $$
# \begin{align*}
# 0 = \max_{\tilde e}  \min_h \min_{\omega_j, \sum_{\ell =1}^L \omega_\ell  = 1}
# & - \delta \phi_m(y)    +  \eta \log \tilde e    \cr
# & + \frac {d \phi_m(y)}{d y} {\tilde e}  \varsigma \cdot h  + {\frac {(\eta - 1)} \delta }\left[\gamma_1 +  \gamma_2 y + \gamma_3^m (y- {\overline y} ) \right] {\tilde e} \varsigma \cdot h + {\frac {\xi_r} 2} h'h \cr 
# & + \frac {d \phi_m(y)}{d y}  \sum_{\ell=1}^L \omega_\ell  \theta_\ell {\tilde e} + {\frac 1 2} \frac {d^2 \phi_m(y)}{(dy)^2} |\varsigma|^2 \tilde e^2  \cr
# &+ {\frac {(\eta - 1)} \delta}  \left( \left[ \gamma_1 + \gamma_2 y + \gamma_3^m (y - \overline y) \right]   \sum_{\ell=1}^L \omega_\ell \theta_\ell {\tilde e} + {\frac 1 2} (\gamma_2 + 
# \gamma_3^m) |\varsigma|^2 \tilde e^2 \right) \cr
# &+ \xi_a \sum_{\ell = 1}^L \omega_\ell \left( \log \omega_\ell - \log \pi_\ell \right).
# \end{align*}
# $$

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import IFrame
from src.model import solve_hjb_y, solve_hjb_y_jump
from src.utilities import find_nearest_value, solve_post_jump
from src.simulation import simulate_jump, no_jump_simulation
import pickle


# In[2]:


# Preference
η = 0.032
δ = 0.01

# Climate sensitivity
θ_list = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0] / 1000.
πc_o = np.ones_like(θ_list) / len(θ_list)

# Damage functions
σ_y = 1.2 * np.mean(θ_list)
y_underline = 1.5
y_bar = 2.
γ_1 = 1.7675 / 10000
γ_2 = 0.0022 * 2
γ_3 = np.linspace(0., 1. / 3, 20)
πd_o = np.ones_like(γ_3) / len(γ_3)

# capital evolution
α = 0.115
i_over_k = 0.09
K0 = 85 / α

# state variable
y_step = .01
y_grid_long = np.arange(0., 5., y_step)
y_grid_short = np.arange(0., 2.1 + y_step, y_step)
n_bar = find_nearest_value(y_grid_long, y_bar) 


# In[3]:


# # Prepare ϕ conditional on low, high, extreme damage
# v_list = {}
# e_tilde_list = {}
# for ξ_r, ξ_a in [(100_000, 100_000), (5., 0.01), (1., 0.01), (0.3, 0.01)]:
#     model_args_list = []
#     for γ_3_m in γ_3:
#         model_arg = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_list, πc_o, ξ_r, ξ_a)
#         model_args_list.append((y_grid_long, model_arg, None, 1., 1e-8, 5_000, False))
#     v_list[ξ_r], e_tilde_list[ξ_r] = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, model_args_list)


# In[4]:


#pickle.dump(v_list, open("v_list", "wb"))
#pickle.dump(e_tilde_list, open("e_tilde_list", "wb"))
v_list = pickle.load(open("v_list", "rb"))
e_tilde_list = pickle.load(open("e_tilde_list", "rb"))


# ### 4.1.2 Pre-jump value function
# 
# The pre-jump value function has a similar structure with two exceptions:
#   -  we include the intensity function discussed earlier and 
#   -  we introduce robustness concerns for both the 
# intensity and distribution over the alternative $\gamma_3^m$ coefficients.  
# 
# Given these modifications, we include:
# 
# 
# $$
# \mathcal J (y) \sum_{m=1}^M g_m \pi_m \left[ \phi_m(\overline y) - \phi(y) \right]
# + \xi_r {\mathcal J}(y)  \sum_{m=1}^M \pi_m \left( 1 - g_m + g_m \log g_m \right)\pi_m 
# $$
# 
# in the HJB and solve for pre-jump value function $\phi(y)$ on $[0, \overline{y}]$:
# 
# $$
# \begin{align*}
# 0 = \max_{\tilde e}  \min_h \min_{\omega_j, \sum_{\ell =1}^L \omega_\ell  = 1} \min_{g_m \geqslant 0}
# & - \delta \phi(y)    +  \eta \log \tilde e    \cr
# & + \frac {d \phi(y)}{d y} {\tilde e}  \varsigma \cdot h  + {\frac {(\eta - 1)} \delta }\left[\gamma_1 +  \gamma_2 y) \right] {\tilde e} \varsigma \cdot h + {\frac {\xi_r} 2} h'h \cr 
# & + \frac {d \phi(y)}{d y}  \sum_{\ell=1}^L \omega_\ell  \theta_\ell {\tilde e} + {\frac 1 2} \frac {d^2 \phi(y)}{(dy)^2} |\varsigma|^2 \tilde e^2  \cr
# &+ {\frac {(\eta - 1)} \delta}  \left( \left[ \gamma_1 + \gamma_2 y\right]   \sum_{\ell=1}^L \omega_\ell \theta_\ell {\tilde e} + {\frac 1 2} \gamma_2  |\varsigma|^2 \tilde e^2 \right) \cr
# &+ \xi_a \sum_{\ell = 1}^L \omega_\ell \left( \log \omega_\ell - \log \pi_\ell \right)\cr
# &+ \mathcal J (y) \sum_{m=1}^M g_m \pi_m \left[ \phi_m(\overline y) - \phi(y) \right]
# + \xi_r {\mathcal J}(y)  \sum_{m=1}^M \pi_m \left( 1 - g_m + g_m \log g_m \right)\pi_m 
# \end{align*}
# $$

# In[5]:


# pre_jump_res = {}
# ξ_r_list = [100_000, 5., 1., 0.3]
# for ξ_r_i in ξ_r_list:
#     ϕ_list = v_list[ξ_r_i]
#     certainty_equivalent = -ξ_r_i * np.log(
#         np.average(
#             np.exp(-1. / ξ_r_i * np.array(ϕ_list)), axis=0, weights=πd_o))
#     # Change grid from 0-4 to 0-2
#     ϕ_i = np.array(
#         [temp[n_bar] * np.ones_like(y_grid_short) for temp in ϕ_list])

#     # Compute ϕ with jump (impose boundary condition)
#     if ξ_r_i == 100_000:
#         ξ_a = 100_000
#     else:
#         ξ_a = 0.01
#     model_args = (η, δ, σ_y, y_underline, y_bar, γ_1, γ_2, γ_3, θ_list, πc_o, ϕ_i, πd_o,
#                   ξ_r_i, ξ_r_i, ξ_a)
#     model_res = solve_hjb_y_jump(y_grid_short,
#                                  model_args,
#                                  v0=None,
#                                  ϵ=1.,
#                                  tol=1e-8,
#                                  max_iter=5_000,
#                                  print_iteration=False)
#     simulation_res = no_jump_simulation(model_res, dt=1/4)
#     pre_jump_res[ξ_r_i] = dict(model_res=model_res,
#                            simulation_res=simulation_res,
#                            certainty_equivalent=certainty_equivalent)


# In[6]:


#pickle.dump(pre_jump_res, open("pre_jump_res", "wb"))
pre_jump_res = pickle.load(open("pre_jump_res", "rb"))


# ### Robust adjustment to climate model uncertainty

# In[7]:


from src.plots import plot5
plot5(pre_jump_res).write_html('plot5_pre_jump_res.html')
IFrame(src ='plot5_pre_jump_res.html', width=700, height=500)


# ### Robust adjustments to damage function uncertainty
# 

# In[8]:


from src.plots import plot6
plot6(pre_jump_res).write_html('plot6_pre_jump_res.html')
IFrame(src ='plot6_pre_jump_res.html', width=700, height=500)


# In[9]:


from src.plots import plot7
plot7(pre_jump_res).write_html('plot7_pre_jump_res.html')
IFrame(src ='plot7_pre_jump_res.html', width=700, height=500)


# ### Emission and anomaly trajectories
# 
# The figure shows emission as a function of temperature anomaly.
# 
# For $\underline y = 1.5$ and $\overline y = 2$, and $\underline y = 1.75$ and $\overline y = 2.25$

# In[10]:


# Repeat for 1.75 - 2.25
y_underline_higher = 1.75
y_bar_higher = 2.25
# state variable
y_step = .01
y_grid_short_2 = np.arange(0., 2.3 + y_step, y_step)
n_bar = find_nearest_value(y_grid_long, y_bar_higher)


# In[11]:


# # post jump value functions
# v175_list = {}
# e175_tilde_list = {}
# for ξ_r, ξ_a in [(100_000, 100_000), (5., 0.01), (1., 0.01), (0.3, 0.01)]:
#     model_args_list = []
#     for γ_3_m in γ_3:
#         model_arg = (η, δ, σ_y, y_bar_higher, γ_1, γ_2, γ_3_m, θ_list, πc_o,
#                      ξ_r, ξ_a)
#         model_args_list.append(
#             (y_grid_long, model_arg, None, 1., 1e-8, 5_000, False))
#     v175_list[ξ_r], e175_tilde_list[ξ_r] = solve_post_jump(
#         y_grid_long, γ_3, solve_hjb_y, model_args_list)

# # pre jump value function


# In[12]:


# pre_jump175_res = {}
# ξ_r_list = [100_000, 5., 1., 0.3]
# for ξ_r_i in ξ_r_list:
#     ϕ_list = v175_list[ξ_r_i]
#     certainty_equivalent = -ξ_r_i * np.log(
#         np.average(
#             np.exp(-1. / ξ_r_i * np.array(ϕ_list)), axis=0, weights=πd_o))
#     # Change grid from 0-4 to 0-2
#     ϕ_i = np.array(
#         [temp[n_bar] * np.ones_like(y_grid_short_2) for temp in ϕ_list])

#     # Compute ϕ with jump (impose boundary condition)
#     if ξ_r_i == 100_000:
#         ξ_a = 100_000
#     else:
#         ξ_a = 0.01
#     model_args = (η, δ, σ_y, y_underline_higher, y_bar_higher, γ_1, γ_2, γ_3,
#                   θ_list, πc_o, ϕ_i, πd_o, ξ_r_i, ξ_r_i, ξ_a)
#     model_res = solve_hjb_y_jump(y_grid_short_2,
#                                  model_args,
#                                  v0=None,
#                                  ϵ=1.,
#                                  tol=1e-8,
#                                  max_iter=5_000,
#                                  print_iteration=False)
#     simulation_res = no_jump_simulation(model_res, dt=1 / 4)
#     pre_jump175_res[ξ_r_i] = dict(model_res=model_res,
#                                   simulation_res=simulation_res,
#                                   certainty_equivalent=certainty_equivalent)


# In[13]:


#pickle.dump(v175_list, open("v175_list", "wb"))
#pickle.dump(e175_tilde_list, open("e175_tilde_list", "wb"))
#pickle.dump(pre_jump175_res, open("pre_jump175_res", "wb"))
pre_jump175_res = pickle.load(open("pre_jump175_res", "rb"))


# In[14]:


from src.plots import plot89

plot8 = plot89(pre_jump_res, y_grid_short_2, y_underline)
plot8.update_layout(
    title=
    r"""Figure 8 : Emissions as a function of the temperature anomaly. <br>
    The thresholds are y̲ = 1.5 and ȳ = 2.0.
   """
)
plot8.write_html('plot8.html')
IFrame(src ='plot8.html', width=700, height=500)


# In[15]:


plot9 = plot89(pre_jump175_res, y_grid_short_2, y_underline_higher)
plot9.update_layout(
    title=
    r"""Figure 9 : Emissions as a function of the temperature anomaly. <br>
    The thresholds are y̲ = 1.75 and ȳ = 2.25.
   """
)
plot9.write_html('plot9.html')
IFrame(src ='plot9.html', width=700, height=500)


# The figure shows $\log SCC$ as a function of temperature anomaly:
# 
# $$
# \log SCC = \log C_0 - \log N - \log E + \log \eta - \log (1 - \eta)
# $$

# In[16]:


from src.plots import plot1011
args_scc = (α, η, i_over_k, K0, γ_1, γ_2)
plot1011(pre_jump_res, pre_jump175_res, y_grid_short_2, y_underline, y_underline_higher, args_scc).write_html('plot1011.html')
IFrame(src ='plot1011.html', width=700, height=500)


# ### Temperature anomalies
# 

# In[17]:


import plotly.graph_objects as go
from src.simulation import EvolutionState
from scipy import interpolate


# In[18]:


e_grid_1 = pre_jump_res[1]["model_res"]["e_tilde"]
e_func_pre_damage = interpolate.interp1d(y_grid_short, e_grid_1)
e_grid_long_1 = e_tilde_list[1]
e_func_post_damage = [interpolate.interp1d(y_grid_long, e_grid_long_1[i]) for i in range(len(γ_3))]

# start simulation
e0 = 0
y0 = 1.1
temp_anol0 = 1.1
y_underline = 1.5
y_overline = 2.
initial_state = EvolutionState(t=0,
                               prob=1,
                               damage_jump_state='pre',
                               damage_jump_loc=None,
                               variables=[e0, y0, temp_anol0],
                               y_underline=y_underline,
                               y_overline=y_overline)

fun_args = (e_func_pre_damage, e_func_post_damage)

T = 410
sim_res = []
temp_anols = []
probs = []
damage_locs = []
sim_res.append([initial_state])
for i in range(T):
    if i == 0:
        states = initial_state.evolve(np.mean(θ_list), fun_args)
    else:
        temp = []
        for state in states:
            temp += state.evolve(np.mean(θ_list), fun_args)
        states = temp
    tempanol_t = []
    probs_t = []
    damage_loc_t = []
    for state in states:
        tempanol_t.append( state.variables[2] )
        probs_t.append( state.prob )
        damage_loc_t.append( state.damage_jump_loc )

    temp_anols.append(tempanol_t)
    probs.append(probs_t)
    damage_locs.append(damage_loc_t)
    sim_res.append(states)


# In[19]:


γ_3_ems_spline=interpolate.make_interp_spline(γ_3, [state.variables[0] for state in sim_res[233][:20]])
γ_3_interp = np.linspace(0,1/3, 3000)
ems_interp = γ_3_ems_spline(γ_3_interp)

fig = go.Figure()
fig.add_trace(go.Scatter(x=γ_3_interp, y=ems_interp))
fig.update_xaxes(range=[-0.01, 1./3], showline=True, title=r"$\gamma_3$")
fig.update_yaxes(title="Emission", range=[0, 10])
fig.write_html('fig.html')
IFrame(src ='fig.html', width=700, height=500)

