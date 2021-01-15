
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

excel_file <- "E:/Calcium Imaging/stats/stats summary/transition.xlsx"
wb <- createWorkbook()

#####################################################################
################### LOAD DATA FOR TRANSITION STUFF ##################
#####################################################################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/transition/transition_state_modulation_index.xlsx")
data = data[complete.cases(data),]
condition <- factor(data$condition)
mouse <- factor(data$animal)
time <- factor(data$time)
neuronID <- factor(data$neuronID)
nPeaks <- data$n_peaks_mod
amplitude <- data$height.median_mod
decay <- data$decay_isol_mod

########### number of peaks ###########

model <- lmer(nPeaks ~ condition*time + (1 | mouse) + (1 | neuronID)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model transition nPeaks")
writeData(wb, sheet = "model transition nPeaks", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ time | condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "nPeaks means")
writeData(wb, sheet = "nPeaks means", posthoc$emmeans)
addWorksheet(wb, "nPeaks contrasts")
writeData(wb, sheet = "nPeaks contrasts", posthoc$contrasts)

########### amplitude ###########

model <- lmer(amplitude ~ condition*time + (1 | mouse) + (1 | neuronID)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model transition amplitude")
writeData(wb, sheet = "model transition amplitude", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ time | condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means")
writeData(wb, sheet = "amplitude means", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts")
writeData(wb, sheet = "amplitude contrasts", posthoc$contrasts)

########### decay ###########

model <- lmer(decay ~ condition*time + (1 | mouse) + (1 | neuronID)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model transition decay")
writeData(wb, sheet = "model transition decay", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ time | condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means")
writeData(wb, sheet = "decay means", posthoc$emmeans)
addWorksheet(wb, "decay contrasts")
writeData(wb, sheet = "decay contrasts", posthoc$contrasts)

##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
