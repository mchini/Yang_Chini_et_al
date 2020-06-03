
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

########################################################################
################## LOAD DATA FOR STANDARD TRANSIENTS ###################
########################################################################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/robustness/standard.xlsx")
condition <- factor(data$condition)
recording <- factor(data$recording)
mouse <- factor(data$mouse)
integral <- data$log_integral
st_dev <- data$log_st_dev

excel_file <- "E:/Calcium Imaging/stats/stats summary/robustness.xlsx"
wb <- createWorkbook()

##################### INTEGRAL #####################

model <- lmer(integral ~ condition + (1 | mouse) + (1 | recording)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model integral")
writeData(wb, sheet = "model integral", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "integral means")
writeData(wb, sheet = "integral means", posthoc$emmeans)
addWorksheet(wb, "integral contrasts")
writeData(wb, sheet = "integral contrasts", posthoc$contrasts)

##################### STANDARD DEVIATION #####################

model <- lmer(st_dev ~ condition + (1 | mouse) + (1 | recording)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model st_dev")
writeData(wb, sheet = "model st_dev", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "st_dev means")
writeData(wb, sheet = "st_dev means", posthoc$emmeans)
addWorksheet(wb, "st_dev contrasts")
writeData(wb, sheet = "st_dev contrasts", posthoc$contrasts)
remove('posthoc') # too large for memory


########################################################################
########################## LOAD DATA FOR Df/f ##########################
########################################################################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/robustness/dF_over_F.xlsx")
condition <- factor(data$condition)
recording <- factor(data$recording)
mouse <- factor(data$mouse)
integral <- data$log_integral
st_dev <- data$log_st_dev

##################### INTEGRAL #####################

model <- lmer(integral ~ condition + (1 | mouse) + (1 | recording)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model integral dF_over_F")
writeData(wb, sheet = "model integral dF_over_F", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "integral dF_over_F means")
writeData(wb, sheet = "integral dF_over_F means", posthoc$emmeans)
addWorksheet(wb, "integral dF_over_F contrasts")
writeData(wb, sheet = "integral dF_over_F contrasts", posthoc$contrasts)

##################### STANDARD DEVIATION #####################

model <- lmer(st_dev ~ condition + (1 | mouse) + (1 | recording)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model st_dev dF_over_F")
writeData(wb, sheet = "model st_dev dF_over_F", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "st_dev dF_over_F means")
writeData(wb, sheet = "st_dev dF_over_F means", posthoc$emmeans)
addWorksheet(wb, "st_dev dF_over_F contrasts")
writeData(wb, sheet = "st_dev dF_over_F contrasts", posthoc$contrasts)
remove('posthoc') # too large for memory


##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
