# statsmodels GLM example

import statsmodels.api as sm


# print(sm.datasets.star98.NOTE)

data = sm.datasets.star98.load()
data.exog = sm.add_constant(data.exog, prepend=False)

glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())
