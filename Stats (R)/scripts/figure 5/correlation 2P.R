
Sys.setenv(LANG = "en")
rm(list  =  ls())

library("lme4")
library("lmerTest")
library("emmeans")
library("openxlsx")
library("xlsx")

##################### LOAD DATA FOR CORRELATION (REC-BY-REC DATASET - LOW THR) #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/correlation/high thr by rec.xlsx")
condition <- factor(data$condition)
recording <- factor(data$recording)
mouse <- factor(data$mouse)
correlation <- data$AbsTcorrFisher
excel_file <- "E:/Calcium Imaging/stats/stats summary/figureS8.xlsx"
wb <- createWorkbook()

##################### CORRELATION #####################

model <- lmer(correlation ~ condition + (1 | mouse)) # fit the model
anova(model)
addWorksheet(wb, "model correlation")
writeData(wb, sheet = "model correlation", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "correlation means")
writeData(wb, sheet = "correlation means", posthoc$emmeans)
addWorksheet(wb, "correlation contrasts")
writeData(wb, sheet = "correlation contrasts", posthoc$contrasts)

##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb=wb, excel_file)

##################### LOAD DATA FOR CORRELATION (REC-BY-REC DATASET - HIGH THR) #####################

rm(list  =  ls())

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/correlation/low thr by rec.xlsx")
condition <- factor(data$condition)
recording <- factor(data$recording)
mouse <- factor(data$mouse)
correlation <- data$AbsTcorrFisher
excel_file <- "E:/Calcium Imaging/stats/stats summary/figure5.xlsx"
wb <- createWorkbook()

##################### CORRELATION #####################

model <- lmer(correlation ~ condition + (1 | mouse)) # fit the model
anova(model)
addWorksheet(wb, "model correlation")
writeData(wb, sheet = "model correlation", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "correlation means")
writeData(wb, sheet = "correlation means", posthoc$emmeans)
addWorksheet(wb, "correlation contrasts")
writeData(wb, sheet = "correlation contrasts", posthoc$contrasts)

##################### LOAD DATA FOR QUARTILE STUFF #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/correlation/df_qrt.xlsx")
condition <- factor(data$condition)
recording <- factor(data$recording)
mouse <- factor(data$mouse)
qrt1 <- data$`1st_qrt`
qrt4 <- data$`4th_qrt`


##################### 1st QUARTILE #####################

model <- lmer(qrt1 ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite"s method)
addWorksheet(wb, "model 1st quartile")
writeData(wb, sheet = "model 1st quartile", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "1st quartile means")
writeData(wb, sheet = "1st quartile means", posthoc$emmeans)
addWorksheet(wb, "1st quartile contrasts")
writeData(wb, sheet = "1st quartile contrasts", posthoc$contrasts)


##################### 4th QUARTILE #####################

model <- lmer(qrt4 ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite"s method)
addWorksheet(wb, "model 4th quartile")
writeData(wb, sheet = "model 4th quartile", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "4th quartile means")
writeData(wb, sheet = "4th quartile means", posthoc$emmeans)
addWorksheet(wb, "4th quartile contrasts")
writeData(wb, sheet = "4th quartile contrasts", posthoc$contrasts)


##################### LOAD DATA FOR TCOEFF STUFF #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/correlation/Tcoeff.xlsx")
condition <- factor(data$Condition)
mouse <- factor(data$Mouse)
Tcoeff10 <- data$Tcoeff10
Tcoeff1000 <- data$Tcoeff1000


##################### Tcoeff 10ms #####################

model <- lmer(Tcoeff10 ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite"s method)
addWorksheet(wb, "model Tcoeff 10ms")
writeData(wb, sheet = "model Tcoeff 10ms", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "Tcoeff 10ms means")
writeData(wb, sheet = "Tcoeff 10ms means", posthoc$emmeans)
addWorksheet(wb, "Tcoeff 10ms contrasts")
writeData(wb, sheet = "Tcoeff 10ms contrasts", posthoc$contrasts)


##################### Tcoeff 1000ms #####################

model <- lmer(Tcoeff1000 ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite"s method)
addWorksheet(wb, "model Tcoeff 1000ms")
writeData(wb, sheet = "model Tcoeff 1000ms", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "Tcoeff 1000ms means")
writeData(wb, sheet = "Tcoeff 1000ms means", posthoc$emmeans)
addWorksheet(wb, "Tcoeff 1000ms contrasts")
writeData(wb, sheet = "Tcoeff 1000ms contrasts", posthoc$contrasts)

##################### LOAD DATA FOR POPULATION COUPLING #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/correlation/quartiles_pop_coupl.xlsx")
condition <- factor(data$condition)
mouse <- factor(data$mouse)
qrt4 <- data$qrt4

##################### 4th quartile POP COUPLING #####################

model <- lmer(qrt4 ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite"s method)
addWorksheet(wb, "model 4th qrt pop coupling")
writeData(wb, sheet = "model 4th qrt pop coupling", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "4th qrt pop coupling means")
writeData(wb, sheet = "4th qrt pop coupling means", posthoc$emmeans)
addWorksheet(wb, "4th qrt pop coupling contrasts")
writeData(wb, sheet = "4th qrt pop coupling contrasts", posthoc$contrasts)


##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb= wb, excel_file)
