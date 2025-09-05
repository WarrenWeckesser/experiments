library(car)

options(digits=12)

set.seed(0)
data <- data.frame(program = rep(c("A", "B", "C"), each = 100),
                   loss = c(runif(100, -1, 3),
                            runif(100, 0, 5),
                            runif(100, 1, 7)))

result = leveneTest(data$loss, data$program, center="mean", trim=0.2)

print(result$F)
print(result$Pr)
