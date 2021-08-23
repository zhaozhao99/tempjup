#!/usr/bin/env python
# coding: utf-8

# # 5 Illustrative economy II: uncertainty decomposition
# 
# This notebook presents compuations in section 8 of the paper.
# 
# 
# An advantage to the more structured approach implemented as smooth ambiguity is that it allows us to "open the hood" so to speak on uncertainty.   We build on the work of {cite:t}`RickeCaldeira:2014` by exploring the relative contributions of uncertainty in the carbon dynamics versus uncertainty in the temperature dynamics.  We depart from their analysis by studying the relative contributions  in the context of a decision  problem and  we include robustness to model misspecification as a third source of uncertainty. This latter adjustment applies primarily to the damage function specification. We continue to use the social cost of carbon as a benchmark for assessing these contributions. We perform these computations using the model developed in the previous section, although the approach we describe is applicable more generally. For the uncertainty decomposition, we hold fixed the control law for  emissions, and  hence also the implied state evolution for damages, and  explore the consequences of imposing constraints on minimization over  the probabilities across the different models.
# 
# Recall that we use climate sensitivity parameters from combinations of 16 models of temperature dynamics and 9 models of carbon dynamics. A parameter $\theta$ corresponds to climate-temperature model pair.  Let $\Theta$ denote the full set of $L = 144$ pairs, and let $P_{j}$ for $j = 1,2,... J$ be a partition of the positive integers up to $L$.  The integer $J$ is set to 9 or 16 depending on whether we target the temperature models or the carbon. For any given such partition, we solve a constrained version of the following minimization problem 
# 
# $$
# \begin{align*}
# \min_{f, \int f(\theta) \pi(d\theta)=1} &
# \left(\frac {\partial  V}{\partial  x }\right) \cdot
# \int \mu(x, a|\theta) f(\theta) \pi(d\theta) \cr
# &  + \xi_a \int f(\theta) \log(f(\theta)) \pi(d\theta)
# \end{align*}
# $$
# 
# by targeting the probabilities assigned to partitions while imposing the benchmark probabilities conditioned on each partition.
# 
# $$
# \begin{align*}
# \min_{{\overline \omega}_j, j=1,2,..., J} &
# \left(\frac {\partial  V}{\partial  x }\right) \cdot
# \sum_{j=1}^J {\overline \omega}_j \sum_{\ell \in P_j}  \left( {\frac {
# \pi_\ell}  {\sum_{\ell \in P_j} \pi_\ell}} \right) \mu(x, a \mid \theta_\ell) \cr
# &  + \xi_a \sum_{j=1}^J {\overline \omega}_j \left(\log {\overline \omega}_j - \log {\overline \pi}_j\right)
# \end{align*}
# $$
# 
# where: ${\overline \pi}_j = {\sum_{\ell \in P_j} \pi_\ell}$ and 
# 
# $$
# \frac {\pi_\ell}  {{\overline \pi}_\ell }  \hspace{.5cm} \ell \in P_j
# $$
# 
# are the baseline conditional probabilities for partition $j$.  We only minimize the probabilities across partitions while imposing the baseline conditional probabilities within a partition.  
# 
# We impose $\xi_r = \infty$ when performing this minimization and let $\xi_a = .01$ as in section 7 in the original paper. We perform additional calculations where we let $\xi_r=1$ and $\xi_a = \infty$ in order to target damage function uncertainty rather than temperature or climate dynamics uncertainty.\footnote{While the robustness adjustment also applies to the climate dynamics, as we saw in the previous section, this adjustment was small relative to the ambiguity adjustment.}  The two states in our problem are $x = (y,n)$, and we look for a value function of the form $V(y,n) = \phi(y) + \frac{(\eta - 1)}{\delta} n$ while imposing that ${\tilde e} = \epsilon(y)$.  For each partition of interest, we construct the corresponding HJB equation that supports this minimization. 
# 
# Since we are imposing the control law for emissions but constraining the minimization, the first-order conditions for emissions will no longer be satisfied. Recall formula the marginal value formula,
# 
# $$
# MV(x) = \frac{\partial U}{\partial e} [x, \psi(x)] + \frac{\partial V}{\partial x}(x) \cdot \frac{\partial \mu}{\partial e}[x, \psi(x)]  + \frac{1}{2}\text{trace}\left[\frac{\partial^2 V}{\partial x \partial x'} \frac{\partial}{\partial e} \Sigma[x, \psi(x]\right],
# $$
# 
# from section 6 in the original paper with adjustments for uncertainty. In the absence of optimality, the net benefit measure $MV(x)$ is not zero with the minimization constraints imposed. Consistent with the SCC computation from the previous section, we use 
# 
# $$
# \begin{align*} 
#  - \frac {\partial V}{\partial x} (x) \cdot {\frac {\partial \mu}{\partial e}} \left[x, \phi(x) \right]  -  {\frac 1 2}  {\rm trace} \left[  \frac {\partial^2 V}{\partial x \partial x'} (x) \frac \partial  {\partial e} \Sigma \left[x, \phi(x)  \right] \right].
# \end{align*}
# $$
# 
# for our cost contributions in the SCC decomposition.  
# 
# We obtain the smallest cost measure when we preclude minimization altogether while solving for the value function and the largest one when we allow for full minimization with $\xi_r = 1$ and $\xi_a = .01.$  We have three intermediate cases corresponding to temperature dynamic uncertainty, climate dynamic uncertainty and damage function uncertainty.  The smallest of these measures corresponds to a full commitment to the baseline probabilities. We form ratios with respect to the smallest measure, take logarithms and multiply by 100 to convert the numbers to percentages. Importantly, we change both probabilities and value functions in this computation.   
# 
# We  report the results in the figure below.
# From this figure, we see that the uncertainty adjustments in valuation account for twenty to thirty percent of the social cost of carbon.  The  contributions from temperature and carbon are essentially constant over time with the temperature uncertainty contribution being substantially larger.  The damage contribution is initially below half the total uncertainty, but this changes to more than half by the time the temperature anomaly reaches the lower threshold of 1.5 degrees Celsius.  
# 
# For our uncertainty decomposition, we compute the logarithm of this expression for alternative partitions of the models.  We start by activating separately uncertainty aversion over 
# <ol style="list-style-type:lower-roman">
#     <li>models of carbon dynamics,</li> 
#     <li>the models of temperature dynamics, and</li> 
#     <li>the models or economic  damages</li>  
# </ol>    
# In each case we report the difference in logarithms between the computation using the baseline probabilities and the solutions from the constrained probability minimizations.  Importantly, we change both probabilities and value functions in this computation.  
# 
# 
# 
# 

# In[1]:


# packages
import pandas as pd
import numpy as np
from src.model import solve_hjb_y, solve_hjb_y_jump, solve_baseline, minimize_g, minimize_π
from src.utilities import find_nearest_value, solve_post_jump
from src.simulation import simulate_me
from IPython.display import IFrame
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import pickle
pyo.init_notebook_mode()


# In[2]:


# preparation
ξ_w = 100_000
ξ_a = 0.01
ξ_r = 1.

ϵ = 5.
η = .032
δ = .01

θ_list = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0]/1000.
πc_o = np.ones_like(θ_list)/len(θ_list)

σ_y = 1.2*np.mean(θ_list)
y_underline = 1.5
y_bar = 2.
γ_1 = 1.7675/10000
γ_2 = .0022*2
γ_3 = np.linspace(0., 1./3, 20)
πd_o = np.ones_like(γ_3)/len(γ_3)

y_step = .01
y_grid_long = np.arange(0., 5., y_step)
y_grid_short = np.arange(0., y_bar+y_step, y_step)
n_bar = find_nearest_value(y_grid_long, y_bar) 

# Uncertainty decomposition
n_temp = 16
n_carb = 9
θ_reshape = θ_list.reshape(n_temp, n_carb)
θ_temp = np.mean(θ_reshape, axis=1)
θ_carb = np.mean(θ_reshape, axis=0)
πc_o_temp = np.ones_like(θ_temp)/len(θ_temp)
πc_o_carb = np.ones_like(θ_carb)/len(θ_carb)


# In[3]:


pre_jump_res = pickle.load(open("pre_jump_res", "rb"))
v_list = pickle.load(open("v_list", "rb"))
e_tilde_list = pickle.load(open("e_tilde_list", "rb"))


# The results are reported below.
# For  comparison we include the analogous computation when we activate an aversion to all three sources of uncertainty.
# 
# 
# With the penalties, $\xi_p = 5$ and $\xi_a = 0.01$, the contributions from temperature are essentially constant
# over time with the temperature uncertainty contribution being substantially larger. The damage
# contribution is initially well below half the total uncertainty, but this changes to more than half
# after one hundred years. It is important to remember that these computations are performed while imposing the planner’s solution for emissions and damages. So called “business-as-usual”
# simulations would change substantially this uncertainty accounting.
# 
# Since the uncertainty components are not “additive,” we explore the joint impacts by parti-
# tioning the uncertainty using the three different pairings of contributions. The results are reported
# in subfigure (b). Not surprisingly, the combination of temperature and damage uncertainty has the biggest impact accounting for about three-fourths of the total uncertainty. In contrast, the combination of temperature and carbon uncertainty accounts for somewhere between one-half and
# one-third of the total uncertainty depending on how many years in the future we look at.
# 
# The quantitative importance of damages will increase as we reduce $\xi_p$. We see the $\xi_p$ setting
# as dictating how much wiggle room a decision maker wants to entertain for the weighting of the
# alternative damage model specifications. For comparison, we set $\xi_p = 0.3$ to match what we used
# previously for $\xi_b$ (click the button with  $\xi_a = 0.01$ and $\xi_p = 0.3$) . With this change, minimizing probabilities are shifted almost entirely to the
# “extreme damage” specification, given us effectly an upper bound on the uncertainty contribution
# to the social cost of carbon. Now the overall uncertainty contribution ranges from thirty to sixty
# percent as shown in subfigure (a) with $\xi_p = 0.3$ and $\xi_a = 0.01$. The damage uncertainty contribution alone accounts for more
# than half of this where as the temperature and climate contributions remain about the same as
# before. Temperature and damage uncertainty taken together account for most of the uncertainty
# as reflected in subfigure (b).

# In[4]:


ems_star = pre_jump_res[1]["model_res"]['e_tilde']
ME_total = η / ems_star


# In[5]:


args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, 100_000, 100_000, σ_y, y_underline)
ME_base, ratio_base = solve_baseline(y_grid_long,
                                     n_bar,
                                     ems_star[:n_bar + 1],
                                     v_list[100_000], 
                                     args,
                                     ϵ=1.,
                                     tol=1e-8,
                                     max_iter=500)

# carbon
print("--------------Carbon-----------------")
args_list_carb = []
for γ_3_m in γ_3:
    args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_carb, πc_o_carb, 100_000, 0.01)
    args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
    args_list_carb.append(args_iter)

ϕ_list_carb, ems_list_carb = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_carb)
args = (δ, η, θ_carb, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, 0.01, 100_000, σ_y, y_underline)
ME_carb, ratiocarb = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_carb, args)

# temperature
print("-------------Temperature--------------")
args_list_temp = []
for γ_3_m in γ_3:
    args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_temp, πc_o_temp, 100_000, 0.01)
    args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
    args_list_temp.append(args_iter)

ϕ_list_temp, ems_list_temp = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_temp)
args = (δ, η, θ_temp, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, 0.01, 100_000, σ_y, y_underline)
ME_temp, ratiotemp = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_temp, args)

# damage
print("-------------------Damage-----------------")
args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 1, 100_000, 100_000, σ_y, y_underline)
ME_dmg, ratiotemp = minimize_g(y_grid_long, n_bar, ems_star[:n_bar + 1], v_list[100_000], args)


# In[6]:


loc_11 = np.abs(y_grid_long - 1.1).argmin()
loc_15 = np.abs(y_grid_long - y_underline).argmin()
ratios = [
    np.log(ME_total[loc_11:loc_15 + 1] / ME_base[loc_11:loc_15 + 1]) * 100,
    np.log(ME_dmg[loc_11:loc_15 + 1] / ME_base[loc_11:loc_15 + 1]) * 100,
    np.log(ME_temp[loc_11:loc_15 + 1] / ME_base[loc_11:loc_15 + 1]) * 100,
    np.log(ME_carb[loc_11:loc_15 + 1] / ME_base[loc_11:loc_15 + 1]) * 100,
]
from src.plots import plot13
plot13(ratios, y_grid_long, y_underline).write_html('plot13.html')
IFrame(src ='plot13.html', width=700, height=500)


# In[ ]:




