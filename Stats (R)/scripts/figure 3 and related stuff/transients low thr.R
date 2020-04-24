
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')
library("xlsx")

# dataset is too large to compute the degrees of freedom with standard lme method 
# you can however ignore the warning about lack of calculations for degrees of freedom, the pvalue is still accurate

##################### LOAD DATA FOR LOW THR DATASET #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/transients/transients low thr.xlsx")
factor(data$condition) -> condition
factor(data$mouse) -> mouse
factor(data$recording) -> recording
data$n_peaks -> n_peaks
data$height -> height
data$decay_isol_10 -> decay
excel_file <- "E:/Calcium Imaging/stats/stats summary/supp_figure3.xlsx"
wb <- createWorkbook()

# in this case the grouping variable is not only mouse but also recording
# before this was not the case because we only had one parameter value per recording, 
# now we have one per neuron

##################### NUMBER OF PEAKS #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | recording)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model number of peaks")
writeData(wb, sheet = "model number of peaks", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "number of peaks means")
writeData(wb, sheet = "number of peaks means", posthoc$emmeans)
addWorksheet(wb, "number of peaks contrasts")
writeData(wb, sheet = "number of peaks contrasts", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | recording)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude")
writeData(wb, sheet = "model amplitude", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means")
writeData(wb, sheet = "amplitude means", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts")
writeData(wb, sheet = "amplitude contrasts", posthoc$contrasts)
remove('posthoc') # too large for memory


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | recording)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay transients")
writeData(wb, sheet = "model decay transients", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison (takes a long time)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay transients means")
writeData(wb, sheet = "decay transients means", posthoc$emmeans)
addWorksheet(wb, "decay transients contrasts")
writeData(wb, sheet = "decay transients contrasts", posthoc$contrasts)
remove('posthoc') # too large for memory

##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
