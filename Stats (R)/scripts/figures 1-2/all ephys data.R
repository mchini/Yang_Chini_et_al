
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

##################### LOAD DATA FOR LFP #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/LFP_power_MI.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
frequency <- factor(data$Frequency)
power <- data$Power
excel_file <- "E:/Calcium Imaging/stats/stats summary/all ephys data.xlsx"
wb <- createWorkbook()


model <- lmer(power ~ condition * frequency + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model LFP power")
writeData(wb, sheet = "model LFP power", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ condition | frequency)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "LFP power means")
writeData(wb, sheet = "LFP power means", posthoc$emmeans)
addWorksheet(wb, "LFP power contrasts")
writeData(wb, sheet = "LFP power contrasts", posthoc$contrasts)


##################### LOAD DATA FOR ACTIVE PERIODS #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/Active_periods_MI.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
time <- factor(data$Time)
active_periods <- data$`%.active.period`


model <- lmer(active_periods ~ time * condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model active periods")
writeData(wb, sheet = "model active periods", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ time | condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "active periods means")
writeData(wb, sheet = "active periods means", posthoc$emmeans)
addWorksheet(wb, "active periods contrasts")
writeData(wb, sheet = "active periods contrasts", posthoc$contrasts)


##################### LOAD DATA FOR PAC #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/PAC_MI.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
frequency <- factor(data$Frequency)
PAC <- data$PAC


model <- lmer(PAC ~ condition * frequency + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model PAC")
writeData(wb, sheet = "model PAC", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ condition | frequency)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "PAC means")
writeData(wb, sheet = "PAC means", posthoc$emmeans)
addWorksheet(wb, "PAC contrasts")
writeData(wb, sheet = "PAC contrasts", posthoc$contrasts)


##################### LOAD DATA FOR FIRING RATE #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/FiringRate_long.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
unit <- factor(data$Unit)
Firing.Rate <- data$Firing.Rate


model <- lmer(Firing.Rate ~ condition + (1 | mouse) + (1 | unit)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model firing rate")
writeData(wb, sheet = "model firing rate", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "firing rate means")
writeData(wb, sheet = "firing rate means", posthoc$emmeans)
addWorksheet(wb, "firing rate contrasts")
writeData(wb, sheet = "firing rate contrasts", posthoc$contrasts)


##################### LOAD DATA FOR ACTIVE UNITS #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/Active_units_long.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
time <- factor(data$Time)
ActiveUnits <- data$ActiveUnits


model <- lmer(ActiveUnits ~ time * condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model active_units")
writeData(wb, sheet = "model active_units", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ time | condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "active_units means")
writeData(wb, sheet = "active_units means", posthoc$emmeans)
addWorksheet(wb, "active_units contrasts")
writeData(wb, sheet = "active_units contrasts", posthoc$contrasts)


##################### LOAD DATA FOR SUA POWER #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/SUA_power_long.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
frequency <- factor(data$Frequency)
power <- data$Power


model <- lmer(power ~ condition * frequency + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model SUA power")
writeData(wb, sheet = "model SUA power", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ condition | frequency)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "SUA power means")
writeData(wb, sheet = "SUA power means", posthoc$emmeans)
addWorksheet(wb, "SUA power contrasts")
writeData(wb, sheet = "SUA power contrasts", posthoc$contrasts)


##################### LOAD DATA FOR PPC #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/PPC_all_conditions_long.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
frequency <- factor(data$Frequency)
unit <- factor(data$Unit)
PPC <- data$PPCval


model <- lmer(PPC ~ condition * frequency + (1 | mouse) + (1 | unit)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model PPC")
writeData(wb, sheet = "model PPC", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ condition | frequency)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "PPC means")
writeData(wb, sheet = "PPC means", posthoc$emmeans)
addWorksheet(wb, "PPC contrasts")
writeData(wb, sheet = "PPC contrasts", posthoc$contrasts)

##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)


##################### LOAD DATA FOR LFP #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/slope.xlsx")
condition <- factor(data$condition)
mouse <- factor(data$mouse)
slope <- data$slope
excel_file <- "E:/Calcium Imaging/stats/stats summary/slope.xlsx"
wb <- createWorkbook()


model <- lmer(slope ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model slope")
writeData(wb, sheet = "model slope", anova(model))

posthoc <- emmeans(model, trt.vs.ctrl ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "slope means")
writeData(wb, sheet = "slope means", posthoc$emmeans)
addWorksheet(wb, "slope contrasts")
writeData(wb, sheet = "slope contrasts", posthoc$contrasts)


##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)







