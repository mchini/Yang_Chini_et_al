
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

excel_file <- "F:/Calcium Imaging/Sleep Stuff/stats/stats summary/sleep.xlsx"
wb <- createWorkbook()

#####################################################################
######################### LOAD DATA FOR LFP #########################
#####################################################################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/LFP/fixed/LFP_Sleep_MI.xlsx")
condition <- factor(data$Group)
mouse <- factor(data$Animal)
frequency <- factor(data$Frequency)
power <- data$LFP.power.MI

model <- lmer(power ~ condition * frequency + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model LFP power")
writeData(wb, sheet = "model LFP power", anova(model))

posthoc <- emmeans(model, pairwise ~ condition | frequency)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "LFP power means")
writeData(wb, sheet = "LFP power means", posthoc$emmeans)
addWorksheet(wb, "LFP power contrasts")
writeData(wb, sheet = "LFP power contrasts", posthoc$contrasts)


#####################################################################
###################### LOAD DATA FOR 1/f slope ######################
#####################################################################

data <- read.xlsx("F:/Calcium Imaging/Sleep Stuff/results/df4stats/slope.xlsx")
condition <- factor(data$Condition)
mouse <- factor(data$animal)
slope <- data$slope

# fit the model (mouse is not included in the model, as it worsens the fit and does not affect the results)
model <- lmer(slope ~ condition + (1 | mouse)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model slope")
writeData(wb, sheet = "model slope", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "slope means")
writeData(wb, sheet = "slope means", posthoc$emmeans)
addWorksheet(wb, "slope contrasts")
writeData(wb, sheet = "slope contrasts", posthoc$contrasts)

#####################################################################
##################### LOAD DATA FOR FIRING RATE #####################
#####################################################################

data <- read.xlsx("F:/Calcium Imaging/Sleep Stuff/results/df4stats/FR.xlsx")
condition <- factor(data$Condition)
mouse <- factor(data$Mouse)
unit <- factor(data$Unit)
Firing.Rate <- data$FR

# fit the model (mouse is not included in the model, as it worsens the fit and does not affect the results)
model <- lmer(Firing.Rate ~ condition + (1 | unit)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model firing rate SUA")
writeData(wb, sheet = "model firing rate SUA", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "firing rate SUA means")
writeData(wb, sheet = "firing rate SUA means", posthoc$emmeans)
addWorksheet(wb, "firing rate SUA contrasts")
writeData(wb, sheet = "firing rate SUA contrasts", posthoc$contrasts)

#####################################################################
################### LOAD DATA FOR FIRING AWA vs MI ##################
#####################################################################

data <- read.xlsx("F:/Calcium Imaging/Sleep Stuff/results/df4stats/MI_FRawa.xlsx")
mouse <- factor(data$mouse)
MI_NREM <- data$MI_NREM
MI_REM <- data$MI_REM
FR_awa <- data$FR_awa

model <- lmer(MI_NREM ~ FR_awa + (1 | mouse)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model FR awa on NREM_MI rate")
writeData(wb, sheet = "model FR awa on NREM_MI rate", anova(model))

posthoc <- emtrends(model, var = 'FR_awa')
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "FR awa on NREM_MI rate")
writeData(wb, sheet = "FR awa on NREM_MI rate", summary(posthoc))

model <- lmer(MI_REM ~ FR_awa + (1 | mouse)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model FR awa on REM_MI rate")
writeData(wb, sheet = "model FR awa on REM_MI rate", anova(model))

posthoc <- emtrends(model, var = 'FR_awa')
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "FR awa on REM_MI rate")
writeData(wb, sheet = "FR awa on REM_MI rate", summary(posthoc))

#####################################################################
######################### LOAD DATA FOR STTC ########################
#####################################################################

data <- read.xlsx("F:/Calcium Imaging/Sleep Stuff/results/df4stats/Tcoeff.xlsx")
condition <- factor(data$Condition)
mouse <- factor(data$Mouse)
units_pair <- factor(data$UnitsPair)
STTC <- as.numeric(data$Tcoeff)

# fit the model (mouse is not included in the model, as it worsens the fit and does not affect the results)
model <- lmer(STTC ~ condition + (1 | units_pair)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model STTC")
writeData(wb, sheet = "model STTC", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "STTC means")
writeData(wb, sheet = "STTC means", posthoc$emmeans)
addWorksheet(wb, "STTC contrasts")
writeData(wb, sheet = "STTC contrasts", posthoc$contrasts)

#####################################################################
###################### LOAD DATA FOR Transients #####################
#####################################################################

data <- read.xlsx("F:/Calcium Imaging/Sleep Stuff/results/df4stats/transients.xlsx")
condition <- factor(data$Condition)
mouse <- factor(data$Mouse)
recording <- factor(data$Recording)
NPeaks <- data$NpeaksL
Height <- data$HeightL

# fit the model for number of peaks
model <- lmer(NPeaks ~ condition + (1 | mouse) + (1 | recording)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model N Peaks")
writeData(wb, sheet = "model N Peaks", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "N Peaks means")
writeData(wb, sheet = "N Peaks means", posthoc$emmeans)
addWorksheet(wb, "N Peaks contrasts")
writeData(wb, sheet = "N Peaks contrasts", posthoc$contrasts)

# fit the model for height of peaks
model <- lmer(Height ~ condition + (1 | mouse) + (1 | recording)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model Height")
writeData(wb, sheet = "model Height", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "Height means")
writeData(wb, sheet = "Height means", posthoc$emmeans)
addWorksheet(wb, "Height contrasts")
writeData(wb, sheet = "Height contrasts", posthoc$contrasts)

#####################################################################
##################### LOAD DATA FOR Correlations ####################
#####################################################################

data <- read.xlsx("F:/Calcium Imaging/Sleep Stuff/results/df4stats/corr by rec.xlsx")
condition <- factor(data$condition)
mouse <- factor(data$mouse)
recording <- factor(data$recording)
corr <- data$corrF

# fit the model for correlations
model <- lmer(corr ~ condition + (1 | mouse) + (1 | recording)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model corr")
writeData(wb, sheet = "model corr", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "corr means")
writeData(wb, sheet = "corr means", posthoc$emmeans)
addWorksheet(wb, "corr contrasts")
writeData(wb, sheet = "corr contrasts", posthoc$contrasts)


##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)

