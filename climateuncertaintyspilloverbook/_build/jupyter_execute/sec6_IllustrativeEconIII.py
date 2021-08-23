#!/usr/bin/env python
# coding: utf-8

# # 6 Illustrative economy III: carbon abatement technology
# 
# While the model posed in section [4](sec4_IllustrativeEconI.ipynb) illustrated how the unfolding of damages should alter policy, the economic model was not designed to confront transitions to fully carbon-neutral economies.  There have been several calls for such transitions with little regard for the role or impact of uncertainty.  We now modify the model to allow for green technology in decades to come.  
# 
# We next consider a technology that is close to the Dynamic Integrated Climate-Economy (DICE) model  of {cite:t}`Nordhaus:2017`. See also {cite:t}`CaiJuddLontzek:2017` and {cite:t}`CaiLontzek:2019` for a stochastic extension (DSICE)  of the DICE model.
# 
# ```{note}
# :class: dropdown
# Among other stochastic components, the DSICE incorporates tipping elements and characterizes the SCC as a stochastic process. From a decision theory perspective, DSICE focuses on risk aversion and intertemporal substitution under an assumption of rational expectations.
# ```
# 
# For our setting, we alter the output equation from our previous specification as follows:
# 
# $$
# \frac {I_t}{K_t} + \frac {C_t} {K_t}  + \frac {J_t} {K_t}  = \alpha 
# $$
# 
# 
# where:
# 
# $$
# \begin{equation}\label{(12)}\tag{12}
#  \frac {J_t} {K_t} =   \left\{ \begin{matrix}   \alpha {\vartheta_t}  \left[ 1  -  \left({\frac {{\mathcal E}_t} { \alpha \lambda_t K_t}}\right)\right]^\theta & \left({\frac {{\mathcal E}_t} {\alpha  K_t}}\right)  \le \lambda_t \cr
# 0 & \left({\frac {{\mathcal E}_t} {\alpha  K_t}}\right)  \ge \lambda_t  \end{matrix} \right.
# \end{equation}
# $$
# 
# To motivate the term $J_t$, express the emissions to capital ratio as:
# 
# $$
# \frac {{\mathcal E}_t} {K_t}  = \alpha \lambda_t  ( 1 - \iota_t ) 
# $$
# 
# where $0 \le \iota_t \le 1$ is *abatement*  at date $t$. The exogenously specified 
# process $\lambda$ gives the emissions to output ratio in the absence of  any abatement.  
# By investing in $\iota_t$,  this ratio can be reduced, but there is a corresponding reduction in output.  Specifically, the output loss is given by:
# 
# $$
# J_t  = \alpha K_t \vartheta (\iota_t)^\theta 
# $$
# 
# Equation (12) follows by solving for abatement $\iota_t$ in terms of emissions.
# 
# ```{note}
# :class: dropdown
# The link to the specification used in {cite:t}`CaiLontzek:2019` is then:
# 
# $$
# \begin{align*}
# \sigma_t & = \lambda_t \cr
# \vartheta_t & = \theta_{
# 1,t} \cr
# \theta & = \theta_2 \cr
# \mu_t & = \iota_t 
# \end{align*}
# $$
# ```
# 
# The planner's preferences are logarithmic over damaged consumption:
# 
# $$
# \log {\widetilde C}_t = \log C_t  - \log N_t = (\log C_t - \log K_t) - \log N_t + \log K_t .
# $$
# 
# In contrast to the previous specification, the planner's value function for this model is no longer additively separable in $(y,k)$, although it remains additively separable in log damages, $n$. 
#  
# 
# 
# ## HJB, post damage jump value functions
# 
# Controls: $(i, e)$ where $i$ is a potential value for $\frac {I_t} {K_t}$ and $e$ is a realization of ${\mathcal E}_t$.  States are $(k, d, y)$ where $k$ is a realization of $\log K_t$, $d$ is a realization of $\log D_t$, and $y$ is the temperature anomaly.  Guess a value function of the form: 
# $\upsilon_d d + \Phi^m(k,y)$.  
# 
# $$
# \begin{align*} 
# 0 = \max_{i,e} \min_{\omega_\ell, \sum_{\ell = 1}^L \omega_\ell = 1 } & - \delta \upsilon_d d  - \delta \Phi^m(k,y) +  \log \left( \alpha - i -  \alpha \overline{\vartheta} \left[ 1 - \left(\frac {e} { \alpha \overline\lambda \exp(k)} \right) \right]^\theta \right) + k - d \cr 
# & + \frac {\partial \Phi^m(k,y)}{\partial k} 
#  \left[ \mu_k    + i   -
# {\frac { \kappa} 2} i^2  -  \frac  {|\sigma_k|^2}  2 + \frac {|\sigma_k|^2} 2  \frac {\partial^2 \Phi^m(k,y)}{\partial k^2}\right]  \cr
# & + \frac {\partial  \Phi^m(k,y)}{\partial y}  \sum_{\ell=1}^L \omega_\ell  \theta_\ell {e} + {\frac 1 2} \frac {\partial^2 \Phi^m(k,y)}{\partial y^2} |\varsigma|^2 e^2  \cr
# & + \upsilon_d \left( \left[ \gamma_1 + \gamma_2 y + \gamma_3^m (y - \overline y) \right]   \sum_{\ell=1}^L \omega_\ell \theta_\ell { e} + {\frac 1 2} (\gamma_2 + 
# \gamma_3^m) |\varsigma|^2  e^2 \right) \cr
# & + \xi_a \sum_{\ell = 1}^L \omega_\ell \left( \log \omega_\ell - \log \pi_\ell \right) 
# \end{align*} 
# $$
# 
# 
# ### First order condition
# 
# Let 
# 
# $$
# mc \doteq  \frac 1 {\left( \alpha - i -   \alpha \overline{\vartheta} \left[ 1 - \left({\frac {e} { \alpha \overline\lambda \exp(k)}}\right) \right]^\theta  \right)}
# $$
# 
# First-order conditions for $i$:
# 
# $$
# - mc + \frac {\partial \Phi^m(k,y)}{\partial k} \left( 1 - \kappa i \right) = 0.  
# $$
# 
# Given $mc$, the first-order conditions for $i$ are affine.  
# 
# First-order conditions for $e$:
# 
# $$
# \begin{align*} 
#  & mc  \left( \frac{\theta {\overline \vartheta}}{ \overline \lambda} \left[1 - \left({\frac {e} { \alpha \overline\lambda \exp(k)}}\right)  \right]^{\theta - 1}\right) \exp(-k)  \cr 
#  &+  \frac {\partial  \Phi^m(k,y)}{\partial y}  \sum_{\ell=1}^L \omega_\ell  \theta_\ell  + \frac {\partial^2 \Phi^m(k,y)}{\partial y^2} |\varsigma|^2 e \cr 
#  & + 
#  \upsilon_d \left( \left[ \gamma_1 + \gamma_2 y + \gamma_3^m (y - \overline y) \right]   \sum_{\ell=1}^L \omega_\ell \theta_\ell  + (\gamma_2 + 
# \gamma_3^m) |\varsigma|^2 e \right) 
# \end{align*}
# $$
# 
# Given $mc$ and $\theta = 3$, the first-order conditions for $e$ are affine. Update $e$ according to the formula above.
# 
# ### Modification: pre tech jump
# 
# Add 
# 
# $$
#     \xi_r \mathcal{I}_g \left(1 - h + h  \log(h) \right) + \mathcal{I}_g h (\Phi^{\text{post tech, m}} - \Phi^m )
# $$
# 
# 
# to the above HJB.
# 
# 
# ### Modification: pre damage jump
# 
# Given $\Phi^m(k, y)$ for $m = 1, 2, \dots, M$, solve $\Phi(k, y)$ with extra elements to the HJB:
# 
# $$
#   \xi_r \mathcal{I}_g  \sum \pi_d^m (1 - g_m + g_m  \log(g_m) ) + \mathcal{I}_g\sum g_m \pi_d^m(\Phi^{\text{post tech, m}} - \Phi^m )
# $$

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import IFrame
arrival = 20
γ_3 = np.linspace(0, 1./3, 20)
πd_o = np.ones_like(γ_3) / len(γ_3)
θ = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0]/1000.
πc_o = np.ones_like(θ) / len(θ)


# In[2]:


#!python "model_final_5.py"
#!python "model_final_7p5.py"
#!python "model_final_2p5.py"
#!python "model_final_baseline.py"


# In[3]:


gt_tech_2p5 = np.load('new_gt_tech_2p5.npy')
gt_tech_5 = np.load('new_gt_tech_5.npy')
gt_tech_7p5 = np.load('new_gt_tech_7p5.npy')

gt_tech_new_2p5 = np.load('new_gt_tech_new_2p5.npy')
gt_tech_new_5 = np.load('new_gt_tech_new_5.npy')
gt_tech_new_7p5 = np.load('new_gt_tech_new_7p5.npy')

dmg_intensity_distort_2p5 = np.load('new_dmg_intensity_distort_2.5.npy')
dmg_intensity_distort_5 = np.load('new_dmg_intensity_distort_5.0.npy')
dmg_intensity_distort_7p5 = np.load('new_dmg_intensity_distort_7.5.npy')
dmg_intensity_distort_baseline = np.load('new_dmg_intensity_distort_baseline.npy')

intensity_dmg_2p5 = np.load('new_intensity_dmg_2p5.npy')
intensity_dmg_5 = np.load('new_intensity_dmg_5.npy')
intensity_dmg_7p5 = np.load('new_intensity_dmg_7p5.npy')
intensity_dmg_baseline = np.load('new_intensity_dmg_baseline.npy')

πct = np.load("πct.npy")
distorted_damage_probs = np.load("distorted_damage_probs.npy")


# In[4]:


distorted_tech_intensity_first_2p5 = gt_tech_2p5 * 1/arrival
distorted_tech_intensity_first_5 = gt_tech_5 * 1/arrival
distorted_tech_intensity_first_7p5 = gt_tech_7p5 * 1/arrival
tech_intensity_first = np.ones_like(distorted_tech_intensity_first_2p5) * 1/arrival

distorted_tech_intensity_second_2p5 = gt_tech_new_2p5 * 1/arrival
distorted_tech_intensity_second_5 = gt_tech_new_5 * 1/arrival
distorted_tech_intensity_second_7p5 = gt_tech_new_7p5 * 1/arrival
tech_intensity_second = np.ones_like(distorted_tech_intensity_second_2p5) * 1/arrival

distorted_dmg_intensity_2p5 = dmg_intensity_distort_2p5*intensity_dmg_2p5
distorted_dmg_intensity_5 = dmg_intensity_distort_5*intensity_dmg_5
distorted_dmg_intensity_7p5 = dmg_intensity_distort_7p5*intensity_dmg_7p5
distorted_dmg_intensity_baseline = dmg_intensity_distort_baseline*intensity_dmg_baseline

distorted_tech_prob_first_2p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_first_2p5, 0, 0)))[:-1]
distorted_tech_prob_first_5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_first_5, 0, 0)))[:-1]
distorted_tech_prob_first_7p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_first_7p5, 0, 0)))[:-1]
tech_prob_first = 1 - np.exp(-np.cumsum(np.insert(tech_intensity_first, 0, 0)))[:-1]

distorted_tech_prob_second_2p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_second_2p5, 0, 0)))[:-1]
distorted_tech_prob_second_5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_second_5, 0, 0)))[:-1]
distorted_tech_prob_second_7p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_second_7p5, 0, 0)))[:-1]
tech_prob_second = 1 - np.exp(-np.cumsum(np.insert(tech_intensity_second, 0, 0)))[:-1]

distorted_dmg_prob_2p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_dmg_intensity_2p5, 0, 0)))[:-1]
distorted_dmg_prob_5 = 1 - np.exp(-np.cumsum(np.insert(distorted_dmg_intensity_5, 0, 0)))[:-1]
distorted_dmg_prob_7p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_dmg_intensity_7p5, 0, 0)))[:-1]
dmg_prob = 1 - np.exp(-np.cumsum(np.insert(intensity_dmg_baseline, 0, 0)))[:-1]


# In[5]:


from src.plots import plot14
plot14(
    tech_prob_first, distorted_tech_prob_first_7p5, distorted_tech_prob_first_5,distorted_tech_prob_first_2p5,
    tech_prob_second, distorted_tech_prob_second_7p5, distorted_tech_prob_second_5, distorted_tech_prob_second_2p5,
    dmg_prob, distorted_dmg_prob_7p5, distorted_dmg_prob_5, distorted_dmg_prob_2p5
).write_html('plot14.html')
IFrame(src ='plot14.html', width=700, height=500)


# The simulation uses the planner's optimal solution. The left panel shows the distorted jump probabilities for the first technology jump.  The middle panel shows the distorted jump probabilities for the second technology jump. The right panel shows the distorted jump probabilities for the damage function curvature jump. The baseline probabilities for the right panel are computed using the state dependent intensities when we set $\xi_a = \xi_r = \infty.$

# In[6]:


from src.plots import plot15
plot15(θ, γ_3, distorted_damage_probs, πct).write_html('plot15.html')
IFrame(src ='plot15.html', width=700, height=500)


# Figure 15 reports the probability distortions for the damage function and climate sensitivity models. Here, we have imposed $\xi_a = .02$ and $\xi_r = 5.0$. Note that the damage function probability distortions are relatively modest, consistent with our previous discussion.   The climate model distortions, by design, are of similar magnitude as those reported previously in Figure 5.
