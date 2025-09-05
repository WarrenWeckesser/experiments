library(lawstat)

options(digits=12)

set.seed(0)
data <- data.frame(program = rep(c("A", "B", "C"), each = 100),
                   loss = c(runif(100, -1, 3),
                            runif(100, 0, 5),
                            runif(100, 1, 7)))

levene.test(data$loss, data$program, location="trim", trim.alpha=0.2)

write.csv(data, "data.csv", row.names=FALSE, quote=FALSE)
