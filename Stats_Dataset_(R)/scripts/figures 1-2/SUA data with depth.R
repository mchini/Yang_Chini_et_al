
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

##################### LOAD DATA FOR FIRING RATE #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/with depth & 0.1 Hz/Firingrate_Fig_2B.xlsx")
data = data[complete.cases(data),]
condition <- factor(data$Group)
mouse <- factor(data$Animal)
unit <- factor(data$Unit)
Firing.Rate <- data$Firingrate
layer <- data$Layer

excel_file <- "E:/Calcium Imaging/stats/stats summary/SUA data depth.xlsx"
wb <- createWorkbook()

model <- lmer(Firing.Rate ~ condition * layer + (1 | mouse) + (1 | unit)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model firing rate")
writeData(wb, sheet = "model firing rate", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "firing rate means")
writeData(wb, sheet = "firing rate means", posthoc$emmeans)
addWorksheet(wb, "firing rate contrasts")
writeData(wb, sheet = "firing rate contrasts", posthoc$contrasts)


##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
