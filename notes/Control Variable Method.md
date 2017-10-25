---
link: https://en.wikipedia.org/wiki/Control_variates
---
$\newcommand{\var}{\mathrm{Var}}\newcommand{\cov}{\mathrm{Cov}}$

# Control Variable Method (Control Variates)

Funny enough, this is also the basis of how you hedge stock purchases. 

For a random variable X with an expected value $\mu$, you can take another random variable T with expected value t, 
and construct a new variable:
$$
C = X + c (T - t).
$$
Since the original random variable X is an unbiased estimator of $\mu$, we have just created a new unbiased 
estimator $C$ of of $\mu$. Now, by tuning the parameter $c$, we can control the variance of this new estimator $C$.

The variance of $C$ is:
$$
\var(X) + 2 c\,\cov(X, T) + c^2\,\var(T).
$$

Now we can minimize this variance by solving the quadratic equation. The minimum variance is
$$
(1 - \rho^2_{m,\,t})\var(X)
$$ where $$
\rho_{m,\,t} = \cov(m,\,t)
$$