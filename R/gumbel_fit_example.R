# This script is from
#     https://stats.stackexchange.com/questions/71197/
#         usable-estimators-for-parameters-in-gumbel-distribution/71279#71279

#=================================================================================
# Load package
#=================================================================================

library(fitdistrplus)

#=================================================================================
# Define the PDF, CDF and quantile function for the Gumbel distribution
#=================================================================================

dgumbel <- function(x, mu, s){ # PDF
  exp((mu - x)/s - exp((mu - x)/s))/s
}

pgumbel <- function(q, mu, s){ # CDF
  exp(-exp(-((q - mu)/s)))
}

qgumbel <- function(p, mu, s){ # quantile function
  mu-s*log(-log(p))
}

#=================================================================================
# Some data (annual maximum mean daily flows ("annual floods"))
#=================================================================================

flood.data <- c(312, 590, 248, 670, 365, 770, 465, 545, 315, 115, 232, 260, 655, 675,
                455, 1020, 700, 570, 853, 395, 926, 99, 680, 121, 976, 916, 921, 191,
                187, 377, 128, 582, 744, 710, 520, 672, 645, 655, 918, 512, 255, 1126,
                1386, 1394, 600, 950, 731, 700, 1407, 1284, 165, 1496, 809)

#=================================================================================
# Fit the Gumbel distribution using maximum likelihood estimation (MLE)
# Make some diagnostic plots
#=================================================================================

gumbel.fit <- fitdist(flood.data, "gumbel", start=list(mu=5, s=5), method="mle")

summary(gumbel.fit)

# Fitting of the distribution ' gumbel ' by maximum likelihood 
# Parameters : 
#    estimate Std. Error
# mu 471.6864   43.33664
# s  298.8155   32.11813
# Loglikelihood:  -385.1877   AIC:  774.3754   BIC:  778.316 
# Correlation matrix:
#           mu         s
# mu 1.0000000 0.3208292
# s  0.3208292 1.0000000

gofstat(gumbel.fit, discrete=FALSE) # goodness-of-fit statistics

#
# Goodness-of-fit statistics
#                              1-mle-gumbel
# Kolmogorov-Smirnov statistic   0.09956968
# Cramer-von Mises statistic     0.08826106
# Anderson-Darling statistic     0.53360850
#
# Goodness-of-fit criteria
#                                1-mle-gumbel
# Aikake's Information Criterion     774.3754
# Bayesian Information Criterion     778.3160

# Plot the fit

par(cex=1.2, bg="white")
plot(gumbel.fit, lwd=2, col="steelblue")
