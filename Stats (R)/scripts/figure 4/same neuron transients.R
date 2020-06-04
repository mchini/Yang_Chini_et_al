
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

##################### LOAD DATA FOR SAME NEURON ACROSS CONDITIONS #####################

############################################################
###################### ALL CONDITIONS ######################
############################################################

data <- read.xlsx("transients/all_conds in this repo")
condition <- factor(data$condition)
IDneurons <- factor(data$IDneurons)
mouse <- factor(data$mouse)
n_peaks <- data$n_peaks
decay <- data$decay
height <- data$height
excel_file <- "where you want to save the results"
wb <- createWorkbook()

##################### n peaks #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model n peaks")
writeData(wb, sheet = "model n peaks", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "n peaks means")
writeData(wb, sheet = "n peaks means", posthoc$emmeans)
addWorksheet(wb, "n peaks contrasts")
writeData(wb, sheet = "n peaks contrasts", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude")
writeData(wb, sheet = "model amplitude", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means")
writeData(wb, sheet = "amplitude means", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts")
writeData(wb, sheet = "amplitude contrasts", posthoc$contrasts)
remove('posthoc')


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay transients")
writeData(wb, sheet = "model decay transients", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means")
writeData(wb, sheet = "decay means", posthoc$emmeans)
addWorksheet(wb, "decay contrasts")
writeData(wb, sheet = "decay contrasts", posthoc$contrasts)
remove('posthoc')


############################################################
##################### awa-ISOFLURANE #####################
############################################################

data <- read.xlsx("transients/iso_awa in this repo")
condition <- factor(data$condition)
IDneurons <- factor(data$IDneurons)
mouse <- factor(data$mouse)
n_peaks <- data$n_peaks
decay <- data$decay
height <- data$height

##################### n peaks #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model n peaks awa-iso")
writeData(wb, sheet = "model n peaks awa-iso", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "n peaks means awa-iso")
writeData(wb, sheet = "n peaks means awa-iso", posthoc$emmeans)
addWorksheet(wb, "n peaks contrasts awa-iso")
writeData(wb, sheet = "n peaks contrasts awa-iso", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude awa-iso")
writeData(wb, sheet = "model amplitude awa-iso", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means awa-iso")
writeData(wb, sheet = "amplitude means awa-iso", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts awa-iso")
writeData(wb, sheet = "amplitude contrasts awa-iso", posthoc$contrasts)
remove('posthoc')


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay awa-iso")
writeData(wb, sheet = "model decay awa-iso", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means awa-iso")
writeData(wb, sheet = "decay means awa-iso", posthoc$emmeans)
addWorksheet(wb, "decay contrasts awa-iso")
writeData(wb, sheet = "decay contrasts awa-iso", posthoc$contrasts)
remove('posthoc')


############################################################
##################### awa-MMF #####################
############################################################

data <- read.xlsx("transients/fenta_awa in this repo")
condition <- factor(data$condition)
IDneurons <- factor(data$IDneurons)
mouse <- factor(data$mouse)
n_peaks <- data$n_peaks
decay <- data$decay
height <- data$height

##################### n peaks #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model n peaks fenta-awa")
writeData(wb, sheet = "model n peaks fenta-awa", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "n peaks means fenta-awa")
writeData(wb, sheet = "n peaks means fenta-awa", posthoc$emmeans)
addWorksheet(wb, "n peaks contrasts fenta-awa")
writeData(wb, sheet = "n peaks contrasts fenta-awa", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude fenta-awa")
writeData(wb, sheet = "model amplitude fenta-awa", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means fenta-awa")
writeData(wb, sheet = "amplitude means fenta-awa", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts fenta-awa")
writeData(wb, sheet = "amplitude contrasts fenta-awa", posthoc$contrasts)
remove('posthoc') 


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay fenta-awa")
writeData(wb, sheet = "model decay fenta-awa", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means fenta-awa")
writeData(wb, sheet = "decay means fenta-awa", posthoc$emmeans)
addWorksheet(wb, "decay contrasts fenta-awa")
writeData(wb, sheet = "decay contrasts fenta-awa", posthoc$contrasts)
remove('posthoc') 


############################################################
##################### awa-keta #####################
############################################################

data <- read.xlsx("transients/keta_awa in this repo")
condition <- factor(data$condition)
IDneurons <- factor(data$IDneurons)
mouse <- factor(data$mouse)
n_peaks <- data$n_peaks
decay <- data$decay
height <- data$height

##################### n peaks #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model n peaks awa-keta")
writeData(wb, sheet = "model n peaks awa-keta", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "n peaks means awa-keta")
writeData(wb, sheet = "n peaks means awa-keta", posthoc$emmeans)
addWorksheet(wb, "n peaks contrasts awa-keta")
writeData(wb, sheet = "n peaks contrasts awa-keta", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude awa-keta")
writeData(wb, sheet = "model amplitude awa-keta", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means awa-keta")
writeData(wb, sheet = "amplitude means awa-keta", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts awa-keta")
writeData(wb, sheet = "amplitude contrasts awa-keta", posthoc$contrasts)
remove('posthoc') # 


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay awa-keta")
writeData(wb, sheet = "model decay awa-keta", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means awa-keta")
writeData(wb, sheet = "decay means awa-keta", posthoc$emmeans)
addWorksheet(wb, "decay contrasts awa-keta")
writeData(wb, sheet = "decay contrasts awa-keta", posthoc$contrasts)
remove('posthoc') # 


############################################################
##################### ISOFLURANE-MMF #####################
############################################################

data <- read.xlsx("transients/iso_fenta in this repo")
condition <- factor(data$condition)
IDneurons <- factor(data$IDneurons)
mouse <- factor(data$mouse)
n_peaks <- data$n_peaks
decay <- data$decay
height <- data$height

##################### n peaks #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model n peaks iso-MMF")
writeData(wb, sheet = "model n peaks iso-MMF", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "n peaks means iso-MMF")
writeData(wb, sheet = "n peaks means iso-MMF", posthoc$emmeans)
addWorksheet(wb, "n peaks contrasts iso-MMF")
writeData(wb, sheet = "n peaks contrasts iso-MMF", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude iso-MMF")
writeData(wb, sheet = "model amplitude iso-MMF", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means iso-MMF")
writeData(wb, sheet = "amplitude means iso-MMF", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts iso-MMF")
writeData(wb, sheet = "amplitude contrasts iso-MMF", posthoc$contrasts)
remove('posthoc') # 


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay iso-MMF")
writeData(wb, sheet = "model decay iso-MMF", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means iso-MMF")
writeData(wb, sheet = "decay means iso-MMF", posthoc$emmeans)
addWorksheet(wb, "decay contrasts iso-MMF")
writeData(wb, sheet = "decay contrasts iso-MMF", posthoc$contrasts)
remove('posthoc') # 


############################################################
##################### ISOFLURANE-keta #####################
############################################################

data <- read.xlsx("transients/iso_keta in this repo")
condition <- factor(data$condition)
IDneurons <- factor(data$IDneurons)
mouse <- factor(data$mouse)
n_peaks <- data$n_peaks
decay <- data$decay
height <- data$height

##################### n peaks #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model n peaks iso-keta")
writeData(wb, sheet = "model n peaks iso-keta", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "n peaks means iso-keta")
writeData(wb, sheet = "n peaks means iso-keta", posthoc$emmeans)
addWorksheet(wb, "n peaks contrasts iso-keta")
writeData(wb, sheet = "n peaks contrasts iso-keta", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude iso-keta")
writeData(wb, sheet = "model amplitude iso-keta", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means iso-keta")
writeData(wb, sheet = "amplitude means iso-keta", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts iso-keta")
writeData(wb, sheet = "amplitude contrasts iso-keta", posthoc$contrasts)
remove('posthoc') 


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay iso-keta")
writeData(wb, sheet = "model decay iso-keta", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means iso-keta")
writeData(wb, sheet = "decay means iso-keta", posthoc$emmeans)
addWorksheet(wb, "decay contrasts iso-keta")
writeData(wb, sheet = "decay contrasts iso-keta", posthoc$contrasts)
remove('posthoc') 


############################################################
##################### keta-MMF #####################
############################################################

data <- read.xlsx("transients/keta_fenta in this repo")
condition <- factor(data$condition)
IDneurons <- factor(data$IDneurons)
mouse <- factor(data$mouse)
n_peaks <- data$n_peaks
decay <- data$decay
height <- data$height

##################### n peaks #####################

model <- lmer(n_peaks ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model n peaks keta-MMF")
writeData(wb, sheet = "model n peaks keta-MMF", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "n peaks means keta-MMF")
writeData(wb, sheet = "n peaks means keta-MMF", posthoc$emmeans)
addWorksheet(wb, "n peaks contrasts keta-MMF")
writeData(wb, sheet = "n peaks contrasts keta-MMF", posthoc$contrasts)

##################### HEIGHT OF TRANSIENTS #####################

model <- lmer(height ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model amplitude keta-MMF")
writeData(wb, sheet = "model amplitude keta-MMF", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison 
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "amplitude means keta-MMF")
writeData(wb, sheet = "amplitude means keta-MMF", posthoc$emmeans)
addWorksheet(wb, "amplitude contrasts keta-MMF")
writeData(wb, sheet = "amplitude contrasts keta-MMF", posthoc$contrasts)
remove('posthoc') 


##################### DECAY OF TRANSIENTS #####################

model <- lmer(decay ~ condition + (1 | mouse) + (1 | IDneurons)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model decay keta-MMF")
writeData(wb, sheet = "model decay keta-MMF", anova(model))

posthoc <- emmeans(model, pairwise ~ condition) # post-hoc comparison
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "decay means keta-MMF")
writeData(wb, sheet = "decay means keta-MMF", posthoc$emmeans)
addWorksheet(wb, "decay contrasts keta-MMF")
writeData(wb, sheet = "decay contrasts keta-MMF", posthoc$contrasts)
remove('posthoc')

##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
