# Policy Gradient
Policy gradient methods, different from value function approximators, approximate policy $\pi(s_t)$. Generally, the objective of a policy is 
to maximize total rewards of episodes. If a policy is represented by a neural network with parameter $\theta$, and the objective is represented 
by the maximization of function U, and both of them are parameterized as $\pi(s_t; \theta) \thicksim \pi(a_t \vert s_t; \theta)$ and $U(\theta)$, 
the goal and the approach of policy gradient methods are 

$${\begin{align}
  \max_\theta U(\theta) &= \max_\theta E_{\tau \thicksim P_\pi(\tau)}[R(\tau)] \notag &\\
  &= \max_\theta E_{\tau \thicksim P(\tau; \theta)}[R(\tau)] \tag{goal} &\\
  &\text{where } \tau = \{s_t, a_t\}_{t=0}^{T-1} \cup \{s_T\}, P(\tau; \theta) \text{ is the probability of the episode } \tau \notag &\\
  \theta^\prime &= \theta + \alpha \cdot \bigtriangledown U(\theta) \tag{approach}
\end{align}}$$

The gradient of total rewards with respect to $\theta$ can be further derived as 

$${\begin{align}
  \bigtriangledown U(\theta) &= \bigtriangledown E_{\tau \thicksim P(\tau; \theta)}[R(\tau)] \notag &\\
  &= \bigtriangledown \sum_\tau P(\tau; \theta) \cdot R(\tau) &\\
  &= \sum_\tau \bigtriangledown P(\tau; \theta) \cdot R(\tau) \notag &\\
  &= \sum_\tau P(\tau; \theta) \cdot \frac{\bigtriangledown P(\tau; \theta)}{P(\tau; \theta)} \cdot R(\tau) \notag &\\ 
  &= \sum_\tau P(\tau; \theta) \cdot \bigtriangledown \ln P(\tau; \theta) \cdot R(\tau) \notag &\\ 
  &= E_{\tau \thicksim P(\tau; \theta)}[\bigtriangledown \ln P(\tau; \theta) \cdot R(\tau)] \notag &\\
  &= E_{\tau \thicksim P(\tau; \theta)}[\bigtriangledown \ln [\prod_{t=0}^{T-1} P(s_{t+1} \vert s_t, a_t) \cdot \pi(a_t \vert s_t; \theta)] \cdot R(\tau)] \notag &\\ 
  &= E_{\tau \thicksim P(\tau; \theta)}[\bigtriangledown [\sum_{t=0}^{T-1} \ln P(s_{t+1} \vert s_t, a_t) + \sum_{t=0}^{T-1} \ln \pi(a_t \vert s_t; \theta)] \cdot R(\tau)] \notag &\\
  &= E_{\tau \thicksim P(\tau; \theta)}[\bigtriangledown [\sum_{t=0}^{T-1} \ln \pi(a_t \vert s_t; \theta)] \cdot R(\tau)] \notag &\\
  &= E_{\tau \thicksim P(\tau; \theta)}[[\sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t \vert s_t; \theta)] \cdot R(\tau)] \notag &\\
  &\approx \frac{1}{N} \cdot \sum_{i=1}^{N} [\sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta)] \cdot R(\tau^i)
\end{align}}$$

Equation (2) considers all rewards, including all rewards caused by actions done before $a_{t+1}$

$${\begin{align}
  \bigtriangledown U(\theta) &\approx \frac{1}{N} \cdot \sum_{i=1}^{N} (\sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta)) \cdot (\sum_{k=1}^T \gamma^{k-1} \cdot r_k^i) \notag &\\
  &= \frac{1}{N} \cdot \sum_{i=1}^{N} (\sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta)) \cdot (\sum_{k=1}^t \gamma^{k-1} \cdot r_k^i + \sum_{k=t+1}^T \gamma^{k-1} \cdot r_k^i) \notag
\end{align}}$$

Rewards $\sum_{k=1}^t \gamma^{k-1} \cdot r_k^i$ are caused by past actions and should not be considered for future action selection. In other 
words, the goal should not be maximizing total rewards but total future rewards at time t. We denote $\hat{g}$ as the total future rewards part 
of $\bigtriangledown U(\theta)$, and it is

$${\begin{align}
  \hat{g} &= \frac{1}{N} \cdot \sum_{i=1}^{N} \sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \cdot \sum_{k=t+1}^T \gamma^{k-1} \cdot r_k^i \notag &\\
  &= \frac{1}{N} \cdot \sum_{i=1}^{N} \sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \cdot \gamma^t \cdot \sum_{k=t+1}^T \gamma^{k-t-1} \cdot r_k^i \notag &\\
  &= \frac{1}{N} \cdot \sum_{i=1}^{N} \sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \cdot \gamma^t \cdot G_t^i \notag
\end{align}}$$

REINFORCE, Monte Carlo policy gradient, updates $\theta$ with a single episode, $N=1$, by

$${\theta_{t+1} = \theta_t + \alpha \cdot \hat{g}}$$

# Advantage Actor-Critic
A problem of REINFORCE algorithm is high variance of $\hat{g}$, which can lead to varied directions of updates to $\theta$ and slow down the 
learning. A large $N$ can improve this problem, but usually a very large $N$ is required. Another solution is to substract a state dependent 
baseline value from $R(\tau)$. Equation (1) and the expected value after the substraction of $b(s_t)$, any state dependent value, are equal as

$${\begin{align}
  &\bigtriangledown \sum_\tau P(\tau; \theta) \cdot [R(\tau) - b(s_t)] \notag &\\
  &= \bigtriangledown \sum_\tau P(\tau; \theta) \cdot R(\tau) - \bigtriangledown \sum_\tau P(\tau; \theta) \cdot b(s_t) \notag &\\
  &= \bigtriangledown \sum_\tau P(\tau; \theta) \cdot R(\tau) - b(s_t) \cdot \bigtriangledown \sum_\tau P(\tau; \theta) \notag &\\
  &= \bigtriangledown \sum_\tau P(\tau; \theta) \cdot R(\tau) - b(s_t) \cdot \bigtriangledown (1) \notag &\\
  &= \bigtriangledown \sum_\tau P(\tau; \theta) \cdot R(\tau) - b(s_t) \cdot 0 \notag &\\
  &= \bigtriangledown \sum_\tau P(\tau; \theta) \cdot R(\tau) \tag{Equation (1)}
\end{align}}$$

In other words, $\hat{g}$ can also be

$${\hat{g} = \frac{1}{N} \cdot \sum_{i=1}^{N} \sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \cdot \gamma^t \cdot (G_t^i - b(s_t))}$$

The choices of $b(s_t)$ are broad. Any value or function that is independent of $\theta$ can be used. When the state value function $v_\pi(s_t)$ 
is the choice, $G_t - v_\pi(s_t)$ measures the action advantage of $a_t$ over the action chosen by $\pi(s_t)$, for $G_t$ estimating $q_\pi(s_t, a_t)$ 
by sampling and $v_\pi(s_t)$ estimating $q_\pi(s_t, \pi(s_t))$. $v_\pi(s_t)$ can also be a function approximator, parameterized as 
$\hat{v}_\pi(s_t; w)$, and algorithms for estimating value function can also be Monte Carlo or TD. 

1. Monte Carlo
$${\begin{align}
  \min_w J(w) &= \min_w \frac{1}{N} \cdot \sum_{i=1}^{N} \sum_{t=0}^{T-1} [G_t^i - \hat{v}_{\pi}(s_t^i; w)]^2 \tag{loss function} &\\
  \hat{g} &= \frac{1}{N} \cdot \sum_{i=1}^{N} \sum_{t=0}^{T-1} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \cdot \gamma^t \cdot (G_t^i - \hat{v}_{\pi}(s_t^i; w)) \tag{policy gradient} &\\
  w_{t+1} &= w_t - \alpha_w \cdot \bigtriangledown J(w) \tag{gradient decent} &\\
  \theta_{t+1} &= \theta_t + \alpha_{\theta} \cdot \hat{g} \tag{gradient ascent}
\end{align}}$$
2. TD(0)
$${\begin{align}
  \min_w J(w) &= \min_w \frac{1}{N} \cdot \sum_{i=1}^{N} [y^i - \hat{v}_{\pi}(s_t^i; w)]^2 \tag{loss function} &\\
  &\text{where } y^i = \begin{cases} r_{t+1}^i &\text{ if } s_{t+1}^i = s_T, &\\
  r_{t+1}^i + \gamma \cdot \hat{v}_{\pi}(s_{t+1}^i; w^\prime) &\text{otherwise}
  \end{cases} \tag{one-step TD} &\\
  \bigtriangledown J(w) &= - \frac{2}{N} \cdot \sum_{i=1}^{N} [(y^i - \hat{v}_{\pi}(s_t^i; w)) \cdot \bigtriangledown \hat{v}_{\pi}(s_t^i; w)] \notag &\\
  &= - \frac{2}{N} \cdot \sum_{i=1}^{N} \delta^i \cdot \bigtriangledown \hat{v}_{\pi}(s_t^i; w) \notag &\\
  &\text{where } \delta^i = y^i - \hat{v}_{\pi}(s_t^i; w) \tag{action advantage} &\\
  \hat{g} &= \frac{1}{N} \cdot \sum_{i=1}^{N} \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \cdot \gamma^t \cdot (y^i - \hat{v}_{\pi}(s_t^i; w)) \notag &\\
  &= \gamma^t \cdot \frac{1}{N} \cdot \sum_{i=1}^{N} \delta^i \cdot \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \tag{policy gradient} &\\
  w_{t+1} &= w_t + \alpha_w \cdot \frac{1}{N} \sum_{i=1}^{N} \delta^i \cdot \bigtriangledown \hat{v}_{\pi}(s_t^i; w) \tag{gradient decent} &\\
  \theta_{t+1} &= \theta_t + \alpha_{\theta} \cdot \gamma^t \cdot \frac{1}{N} \cdot \sum_{i=1}^{N} \delta^i \cdot \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta) \tag{gradient ascent} &\\
  w^\prime &= w_t, \text{ if t mod }  target\_update = 0 \tag{fixed weights} &\\
  &\text{where } target\_update \in N \notag
\end{align}}$$

$\hat{v}_{\pi}(s_t; w)$ and $\pi(a_t \vert s_t; \theta)$ can be two different neural networks or a single neural network with two different 
heads. $\hat{v}_{\pi}(s_t; w)$ is called the critic, and $\pi(a_t \vert s_t; \theta)$ is called the actor.

Here is the Advantage Actor-Critic (A2C) algorithm

1. Initialize $D$ and burn in with $N$ experience tuples by random policy
2. Initialize $\hat{v}(s_t; w)$ and its clone $\hat{v}(s_t; w^\prime)$.
3. Initialize c = 0
4. Initialize target policy $\pi(a_t \vert s_t; \theta)$
5. repeat for training episodes
    - Initialize $s_t = s_0$
    - $\phi_0 = \phi(s_0)$
    - $I = 1$
    - while $s_t \neq s_T$
        - $a_t = \pi(s_t)$
        - take action $a_t$ and observe $r_{t+1}, s_{t+1}$
        - store $(\phi(s_t), a_t, r_{t+1}, \phi(s_{t+1}))$ in $D$
        - sample a mini-batch with size N from $D$
        - calculate $y_i$ and $\delta^i$ for all samples in the mini-batch
        - update $\hat{v}(s_t; w)$ using $RMSProp(\bigtriangledown J(w))$
        - $w = w + \alpha_w \cdot \frac{1}{N} \cdot \sum_{i=1}^{N} \delta^i \cdot \bigtriangledown \hat{v}(s_t^i; w)$ 
        - $\theta = \theta + \alpha_\theta \cdot I \cdot \frac{1}{N} \cdot \sum_{i=1}^{N} \delta^i \cdot \bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta)$
        - c = c + 1
        - replace $w^\prime$ with $w$ if c % target_update = 0
        - $I = \gamma \cdot I$
        - $s_t = s_{t+1}$

During the training, the A2C policy approximator is on-policy and keeps changing along the update of $\theta$. It means that $\pi(a_t \vert s_t; \theta)$ 
is always trained on data from the previous version of the policy. If we approximated off-policy gradients with Importance Sampling on A2C, we 
would need to add the term in the below to equation (2).

$${\prod_{k=1}^t \frac{\pi_\theta(a_k \vert s_k)}{\pi_{\theta_{old}}(a_k \vert s_k)},}$$

which could explode or vanish in the sequence of an episode. A2C uses experience buffers to make the data more stationary. Another approach 
is Asynchronous Advantage Actor-Critic, A3C (Mnih, 2016), which parallelizes the collection of experience and stablizes training without experience 
buffers. 

# Natural Policy Gradient
The training data is generated by the old policy. If the new policy is too different from the old policy, the policy gradients may make bad 
updates to $\theta$, which may not be recovered. If the difference is too small, the training is inefficient.

Recall that 

$${\begin{align}
  U(\theta) &= E_{\tau \thicksim P(\tau; \theta)}[R(\tau)] \notag &\\
  &= \sum_{t=0}^{T-1} E_{(s_t, a_t) \thicksim P(s_t, a_t; \theta)}[R(s_t, a_t)] \notag &\\
  &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta)}[E_{a_t \thicksim \pi(a_t \vert s_t; \theta)}[R(s_t, a_t)]] \notag
\end{align}}$$

With Importance Sampling, 

$${\begin{align}
  U^{IS}(\theta) &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta_{old})} \frac{p_\theta(s_t)}{p_{\theta_{old}}(s_t)} [E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} R(s_t, a_t)]] \notag &\\
  \approx \hat{U}^{IS}(\theta) \notag 
  &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta_{old})} [E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} R(s_t, a_t)]] \notag &\\
  \bigtriangledown \hat{U}^{IS}(\theta) &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta_{old})} [E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\frac{\bigtriangledown \pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} R(s_t, a_t)]] \notag &\\
  &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta_{old})} [E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} \frac{\bigtriangledown \pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta)} R(s_t, a_t)]] \notag &\\
  &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta_{old})} [E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\bigtriangledown \ln \pi(a_t \vert s_t; \theta) \frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} R(s_t, a_t)]] \notag
\end{align}}$$

Instead of returns, we use advantages, $\hat{A}(s_t, a_t; w^{\prime})$

$${\begin{align}
  \hat{U}^{IS}(\theta) &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta_{old})} [E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} \hat{A}(s_t, a_t; w^{\prime})]] \notag &\\
  &= \hat{E}_t [\frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} \hat{A}(s_t, a_t; w^{\prime})] &\\
  \bigtriangledown \hat{U}^{IS}(\theta) &= \sum_{t=0}^{T-1} E_{s_t \thicksim P(s_t; \theta_{old})} [E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\bigtriangledown \ln \pi(a_t \vert s_t; \theta) \frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} \hat{A}(s_t, a_t; w^{\prime})]] \notag &\\
  &= \hat{E}_t [\bigtriangledown \ln \pi(a_t \vert s_t; \theta) \frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} \hat{A}(s_t, a_t; w^{\prime})]
\end{align}}$$

Let us assume $d$ is the distance between $\theta_{new}$ and $\theta_{old}$, the update vector in the parameter space. To make $d$ as big as 
the training can still work well, we want to find out the optimal distance $d^\star$

$${\begin{align}
  \theta_{new} &= \theta_{old} + d \notag &\\
  d^\star &= \arg \max_{\Vert d \Vert \le \epsilon} \hat{U}^{IS}(\theta_{old} + d) \notag
\end{align}}$$

Natural Policy Gradient method uses KL divergence of $\pi(a_t \vert s_t; \theta_{old})$ and $\pi(a_t \vert s_t; \theta_{old} + d)$ as the distance, 
formulates this problem as an unconstrained optimization problem, and approximates it by the first and 2nd order Taylor expansion of the objective 
and KL divergence respectively.

$${\begin{align}
  \hat{U}^{IS}(\theta) &\approx \hat{U}^{IS}(\theta_{old}) + \bigtriangledown \hat{U}^{IS}(\theta_{old}) \cdot (\theta - \theta_{old}) \tag{1st order} &\\
  KL(\theta) &= KL(\pi(a_t \vert s_t; \theta_{old}), \pi(a_t \vert s_t; \theta)) \notag &\\
  &= E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\ln \frac{\pi(a_t \vert s_t; \theta_{old})}{\pi(a_t \vert s_t; \theta)}] \notag &\\
  &\approx KL(\theta_{old}) + \bigtriangledown KL(\theta_{old}) \cdot (\theta - \theta_{old}) + \frac{1}{2} (\theta - \theta_{old})^T \cdot \bigtriangledown^2 KL(\theta_{old}) \cdot (\theta - \theta_{old}) \tag{2nd order} &\\
  &= \frac{1}{2} (\theta - \theta_{old})^T \cdot \bigtriangledown^2 KL(\theta_{old}) \cdot (\theta - \theta_{old}) \tag{0 and 1st order terms are 0} &\\
  d^\star &= \arg \max_{KL(\pi(a_t \vert s_t; \theta_{old}), \pi(a_t \vert s_t; \theta)) \le \epsilon} \hat{U}^{IS}(\theta_{old} + d) &\\
  &\approx \arg \max_d [\hat{U}^{IS}(\theta_{old} + d) - \lambda (KL(\pi(a_t \vert s_t; \theta_{old}), \pi(a_t \vert s_t; \theta_{old} + d)) - \epsilon)] \notag &\\
  &= \arg \max_d [\hat{U}^{IS}(\theta_{old}) + \bigtriangledown \hat{U}^{IS}(\theta_{old}) \cdot d - \lambda (\frac{1}{2} d^T \cdot \bigtriangledown^2 KL(\theta_{old}) \cdot d - \epsilon)] \notag &\\
  &= \arg \max_d [\bigtriangledown \hat{U}^{IS}(\theta_{old}) \cdot d - \frac{\lambda}{2} d^T \cdot \bigtriangledown^2 KL(\theta_{old}) \cdot d] \notag
\end{align}}$$

By taking partial derivative with respect to $d$ of equation (5) and setting it to 0, we can solve this optimization problem 

$${\begin{align}
  d^\star &= \frac{1}{\lambda} (\bigtriangledown^2 KL(\theta_{old}))^{-1} \cdot \bigtriangledown \hat{U}^{IS}(\theta_{old}) \notag &\\
  &= \frac{1}{\lambda} F(\theta_{old})^{-1} \cdot \bigtriangledown \hat{U}^{IS}(\theta_{old}) \notag &\\
  &\text{where } F(\theta_{old}) \text{ is the Hessian matrix of } KL(\theta_{old}) \text{, or the Fisher Information matrix of } \pi(a_t \vert s_t; \theta_{old}) \notag &\\
  F(\theta_{old}) &= E_{a_t \thicksim \pi(a_t \vert s_t; \theta_{old})}[\bigtriangledown \ln \pi(a_t \vert s_t; \theta_{old}) \cdot (\bigtriangledown \ln \pi(a_t \vert s_t; \theta_{old}))^T] \notag &\\
  &\approx \frac{1}{N} \sum_{i=1, a_t^i \thicksim \pi(a_t \vert s_t; \theta_{old})}^{N}[\bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta_{old}) \cdot (\bigtriangledown \ln \pi(a_t^i \vert s_t^i; \theta_{old}))^T] \notag 
\end{align}}$$

The Natural Policy Gradient picks the direction of $d^{\star}$ and is

$${g_N = F^{-1}(\theta_{old}) \cdot \bigtriangledown \hat{U}^{IS}(\theta_{old})}$$

Back to equation (5), we want to limit the update $d = \alpha_\theta g_N$ within KL divergence equal to $\epsilon$ at most. 

$${\begin{align}
  KL(\theta_{old} + d) &= KL(\theta_{old} + \alpha_\theta g_N) \notag &\\
  &= \frac{1}{2} (\alpha_\theta g_N)^T \cdot F(\theta_{old}) \cdot (\alpha_\theta g_N) \notag &\\
  &= \epsilon \notag
\end{align}}$$

The solution of $\alpha_\theta$ is 

$${\begin{align}
  \alpha_\theta &= \sqrt{\frac{2 \epsilon}{g_N^T F^{-1}(\theta_{old}) g_N}} \notag &\\
  \theta_{new} &= \theta_{old} + \alpha_\theta g_N \notag &\\
  &= \theta_{old} + \sqrt{\frac{2 \epsilon}{g_N^T F^{-1}(\theta_{old}) g_N}} F^{-1}(\theta_{old}) \cdot \bigtriangledown \hat{U}^{IS}(\theta_{old}) \notag
\end{align}}$$

# Proximal Policy Optimization with clipped objective
The calculation of $F^{-1}(\theta_{old})$ is expensive. To avoid the expensive calculation, Proximal Policy Optimization (PPO) with clipped 
objective method formulates the optimization problem with a clipped objective instead.

$${\begin{align}
  r_t(\theta) &= \frac{\pi(a_t \vert s_t; \theta)}{\pi(a_t \vert s_t; \theta_{old})} \notag &\\
  \hat{A}_t &= \hat{A}(s_t, a_t; w^{\prime}) \notag &\\
  \hat{U}^{IS}(\theta) &= \hat{E}_t [r_t(\theta) \hat{A}(s_t, a_t; w^{\prime})] \tag{from equation (3)} &\\
  \theta_{new} &= \arg \max_\theta \text{clipped } [\hat{U}^{IS}(\theta)] \notag &\\
  &= \arg \max_\theta L^{CLIP}(\theta) \notag &\\
  &= \arg \max_\theta \hat{E}_t [min(r_t(\theta) \hat{A}_t, clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t)] \notag &\\
  &\text{where } \epsilon \text{ is a hyperparameter (maybe $\epsilon = 0.2$)} \notag
\end{align}}$$

The clip function is

$${clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) = \begin{cases}
  &r_t(\theta) &\text{if } 1 - \epsilon \lt r_t(\theta) \lt 1 + \epsilon &\\
  &1 - \epsilon &\text{if } r_t(\theta) \le 1 - \epsilon &\\
  &1 + \epsilon &\text{if } 1 + \epsilon \le r_t(\theta) &\\
\end{cases}}$$

When $\hat{A}_t \gt 0$, the clip function returns $(1 + \epsilon) \hat{A}_t$ at most. It can prevent us too greedy, making too much update, 
just because the action advantage looks good. When $\hat{A}_t \lt 0$, the clip function returns $(1 - \epsilon) \hat{A}_t$ at most. It can 
prevent us making too much update such that the negative action won't be taken any more in the new policy.

Here is the PPO with clipped objective algorithm, A2C style

1. Initialize $D$, $K \in [3, 15]$, $M \in [64, 4096]$, $T \in [128, 2048]$ 
2. Initialize $\hat{v}(s_t; w)$ and its clone $\hat{v}(s_t; w^\prime)$.
3. Initialize c = 0
4. Initialize target policy $\pi(a_t \vert s_t; \theta)$
5. repeat for training episodes
    - for actor in range(N):
        - burn in $N \cdot T$ experiences by running $\pi(a_t \vert s_t; \theta)$ for $T$ timesteps
        - Initialize $s_t = s_0$
        - for t in range(T):
            - $a_t = \pi(s_t)$
            - take action $a_t$ and observe $r_{t+1}, s_{t+1}$
            - compute $\hat{A}_t$
            - store $(\phi(s_t), a_t, \pi(a_t \vert s_t; \theta_{old}), r_{t+1}, \phi(s_{t+1}), \hat{A}_t)$ in $D$
            - $s_t = s_{t+1}$
    - update $\hat{v}(s_t; w)$ using $RMSProp(\bigtriangledown J(w))$
    - $w = w + \alpha_w \cdot \frac{1}{N \cdot T} \cdot \sum_{i=1}^{N \cdot T} \delta^i \cdot \bigtriangledown \hat{v}(s_t^i; w)$ 
    - c = c + 1
    - replace $w^\prime$ with $w$ if c % target_update = 0
    - optimize $L^{CLIP}$ with respect to $\theta$, with $K$ epochs and minibatch size $M \le N \cdot T$
    - for epoch in range(K):
        - fill buffer with $D$
        - while buffer is not empty:
            - sample a mini-batch with size $M$ from buffer
            - update $\pi(a_t \vert s_t; \theta)$ using $ADAM(L^{CLIP})$

# Reference
- Carnegie Mellon University, Fragkiadaki, Katerina, et al. 2024. "10-403 Deep Reinforcement Learning" As of 8 November, 2024. https://cmudeeprl.github.io/403website_s24/.
- Sutton, Richard S., and Barto, Andrew G. 2018. Reinforcement Learning - An indroduction, second edition. The MIT Press.
- Mnih, Volodymyr, et al. 2016. Asynchronous Methods for Deep Reinforcement Learning, [arXiv:1602.01783v2](https://arxiv.org/abs/1602.01783)
- Schulman, John, et al. 2017. Proximal Policy Optimization Algorithms, [arXiv:1707.06347v2](https://arxiv.org/abs/1707.06347)
- Wikipedia, n.d., Fisher information. https://en.wikipedia.org/wiki/Fisher_information
- Wikipedia, n.d., Hessian matrix. https://en.wikipedia.org/wiki/Hessian_matrix
- https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl/50663200#50663200
- OpenAI, n.d., https://github.com/openai/baselines/blob/9fa8e1baf1d1f975b87b369a8082122eac812eb1/baselines/ppo1/pposgd_simple.py
- https://github.com/unixpickle/anyrl-py/blob/953ad68d6507b83583e342b3210ed98e03a86a4f/anyrl/algos/ppo.py