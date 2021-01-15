
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

excel_file <- "F:/Calcium Imaging/Sleep Stuff/stats/stats summary/MI_FRawa.xlsx"
wb <- createWorkbook()

#####################################################################
################# LOAD DATA FOR SUA FIRING AWA vs MI ################
#####################################################################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/firing awa + fs/MI_SUA.xlsx")
mouse <- factor(data$Animal)
FR_awa <- data$FiringAwa
condition <- data$Group
MI <- data$MI

model <- lmer(MI ~ FR_awa*condition + (1 | mouse)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model FR awa on SUA MI rate")
writeData(wb, sheet = "model FR awa on SUA MI rate", anova(model))

# this gives you the confidence intervals
posthoc <- emtrends(model, 'condition', var = 'FR_awa')
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "FR awa on SUA MI rate")
writeData(wb, sheet = "FR awa on SUA MI rate", summary(posthoc))

# and this gives you the significance (pvalues)
test(posthoc)
addWorksheet(wb, "FR awa on SUA MI rate pvalues")
writeData(wb, sheet = "FR awa on SUA MI rate pvalues", test(posthoc))

#####################################################################
################# LOAD DATA FOR 2P FIRING AWA vs MI #################
#####################################################################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/firing awa + fs/MI_2P.xlsx")
mouse <- factor(data$Animal)
FR_awa <- data$FiringAwa
condition <- as.factor(data$Group)
MI <- data$MI

model <- lmer(MI ~ FR_awa*condition + (1 | mouse)) 
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model FR awa on 2P MI rate")
writeData(wb, sheet = "model FR awa on 2P MI rate", anova(model))

posthoc <- emtrends(model, 'condition', var = 'FR_awa')
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "FR awa on 2P MI rate")
writeData(wb, sheet = "FR awa on 2P MI rate", summary(posthoc))

addWorksheet(wb, "FR awa on 2P MI rate pvalues")
writeData(wb, sheet = "FR awa on 2P MI rate pvalues", test(posthoc))

##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
