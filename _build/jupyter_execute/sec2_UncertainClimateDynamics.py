#!/usr/bin/env python
# coding: utf-8

# # 2 Uncertain climate dynamics
# 
# 
# ## 2.1  approximation to climate dynamics
# 
# 
# We use exponentially weighted average of each of response functions as coefficients $\{\theta_\ell\}_{\ell=1}^L$ in our computations. 
# The discount rate $\delta=0.01$ and the number of climate models $L = 144$.
# 
# The histogram of those coefficients are represented below:

# In[1]:


import plotly.offline as pyo
from IPython.display import IFrame
pyo.init_notebook_mode()
from src.plots import plot2
plot2().write_html('plot2.html')
IFrame(src ='plot2.html', width=700, height=500)


# ## 2.2 Stochastic climate pulses
# 
# To explore uncertainty, we introduce explicit stochasticity as a precursor to the study of uncertainty.  We capture this randomness in part by an exogenous forcing processes that evolves as:
# 
# $$
# dZ_t = \mu_z(Z_t) dt + \sigma_z(Z_t) dW_t
# $$
# 
# where
# $\{ W_t : t \ge 0\}$  a multivariate standard Brownian motion.  We partition the vector Brownian motion into two subvectors as follows:
# 
# $$
# dW_t = \begin{bmatrix} dW_t^y \cr dW_t^n \cr dW_t^k \end{bmatrix}
# $$
# 
# where the first component consists of the climate change shocks and the second component contains the technology shocks. 
# Consider an emissions "pulse" of the form
# 
# $$
# \left(\iota_y \cdot Z_t \right) {\mathcal E}_t  \left( \theta dt + \varsigma \cdot dW_t^y\right)
# $$
# 
# where ${\mathcal E}_t$ is fossil fuel emissions and $\iota_y \cdot Z = \{ \iota_y \cdot Z_t : t\ge 0\}$ is a positive process which we normalize to have mean one.
# The $\iota_y\cdot Z$-process captures "left out" components of the climate system’s reaction to an emission of ${\mathcal E}_t$ gigatons into the atmosphere while the $\varsigma \cdot dW$ process captures short time scale fluctuations.   
# We will use a positive Feller square root process for the $\iota_y\cdot Z$ process in our analysis.
# 
# 
# 
# 
# Within this framework, we impose the "Matthews' approximation" by making the consequence of the pulse permanent:
# 
# $$
#  dY_t = \mu_y(Z_t, {\mathcal E}_t) dt + \sigma_y(Z_t, {\mathcal E}_t) dW_t^y
# $$
# 
# where
# 
# $$
# \begin{align*}
# \mu_y(z, e) & =  e \left(\iota_y \cdot z \right) \theta   \cr
# \sigma_y(z, e) & = e \left(\iota_y \cdot z \right) \varsigma'
# \end{align*}
# $$
# 
# Throughout, we will use uppercase letters to denote random vector or stochastic processes and lower case letters to denote possible realizations.
# Armed with this "Matthews' approximation", we collapse the climate change uncertainty into the cross-model empirical distribution reported in the figure above. We will eventually introduce uncertainty about $\theta$.
# 
# 
# 
# 
# > **Remark 1**
# >
# > For a more general starting point, let $Y_t$ be a vector used to represent temperature dynamics where the temperature
# impact on damages is the first component of  $Y_t$.
# This state vector evolves according to:
# >
# >
# >$$
# \begin{align*}
# dY_t = \Lambda Y_t dt +   {\mathcal E}_t  \left(\iota_y \cdot Z_t \right)  \left(\Theta dt + \Phi dW_t^y \right)
# \end{align*}
# >$$
# >
# > where $\Lambda$ is a square matrix and $\Theta$ is a column vector.
# > Given an initial condition $Y_0$, the solution for $Y_t$ satisfies:
# >
# > 
# >$$
# Y_t = \exp \left( t \Lambda \right) Y_0 + \int_0^t  \exp\left[ (t-u) \Lambda \right] \left(\iota_y \cdot Z_u \right) {\mathcal E}_u \left(\Theta du + \Phi dW_u^y \right)
# >$$
# >
# > Thus under this specification, the expected future response of $Y$  to a pulse at date zero is:
# > 
# > $$\exp \left( u \Lambda \right) \Theta$$
# >
# > It is the first component of this function that determines the response dynamics.  This generalization allows for multiple exponentials to approximate the pulse responses.  Our introduction of a multiple exponential approximation adapts for example, {cite:t}`Joosetal:2013` and {cite:t}`Pierrehumbert:2014`.
# >
# ```{note}
# :class: dropdown
# See equation (5) of {cite:t}`Joosetal:2013` and  equations (1)-(3) of {cite:t}`Pierrehumbert:2014`. {cite:t}`Pierrehumbert:2014` puts the change in radiative forcing equal to a constant times the logarithm of the ratio of atmospheric $CO_2$ at date $t$ to atmospheric $CO_2$ at baseline date zero. His  Figures 1 and 2 illustrate how an approximation of the Earth System dynamics by three exponentials plus a constant tracks a radiative forcing induced ?by a pulse into the atmosphere at a baseline date from the atmosphere works quite well with half lives of approximately six, sixty five, and four hundred and fifty years.
# ```
# >
# >
# >As an example, we capture the initial rise in the emission responses by the following two-dimensional specification
# >
# >$$\begin{align*} dY_t^1& =  Y_t^2 dt \cr  dY_t^2 & = - \lambda Y_t^2 dt + \lambda  \theta {\mathcal E}_t dt \end{align*}$$
# >
# > which implies the response to a pulse is:
# >
# >$$\theta \left[ 1 - \exp( - \lambda t) \right] {\mathcal E}_0$$
# > 
# > A high value of $\lambda$ implies more rapid convergence to the limiting response $\theta  {\mathcal E}_0$.  This  approximation is intended as a simple representation of the dynamics where the second state variable can be thought of as an exponentially weighted average of current and past emissions.
# >
# ```{note}
# :class: dropdown
# In independent work, {cite:t}`DietzVenmans:2019` and {cite:t}`BarnettBrockHansen:2020` have used such simplified approximations within an explicit economic optimization framework.  The former contribution includes the initial rapid upswing in the impulse response functions.  The latter contribution  abstracts from this.   >{cite:t}`BarnettBrockHansen:2020` instead explore ways to confront uncertainty, broadly-conceived, while using the Matthews approximation.
# ```
# >
# > **Remark 2**
# >
# > The  approximation in {cite:t}`Geoffroy:2013` includes the logarithm of carbon in the atmosphere as argued for by {cite:t}`Arrhenius:1896` which is not directly reflected in the linear approximation to the temperature dynamics that we use.  The pulse experiments from {cite:t}`Joosetal:2013` show a more than proportional change in atmospheric carbon when the pulse size is changed.  It turns out that this is enough to approximately offset the logarithmic {cite:t}`Arrhenius:1896` adjustment so that the long-term temperature response remains approximately proportional for small pulse sizes.  See also {cite:t}`Pierrehumbert:2014` who discusses the approximate offsetting impacts of nonlinearity in temperature and climate dynamics.
