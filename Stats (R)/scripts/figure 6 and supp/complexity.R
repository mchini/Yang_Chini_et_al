
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

##################### LOAD DATA FOR COMPLEXITY #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/figure 6/complexity.xlsx")
condition <- factor(data$condition)
mouse <- factor(data$mouse)
affinity.traces <- data$affinity.traces
pca.traces <- data$pca.traces
tsne.traces <- data$tsne.traces
affinity.spikes <- data$affinity.spikes
tsne.spikes <- data$tsne.spikes
excel_file <- "E:/Calcium Imaging/stats/stats summary/figure6.xlsx"
wb <- createWorkbook()

##################### affinity.traces #####################

model <- lmer(affinity.traces ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model affinity.traces")
writeData(wb, sheet = "model affinity.traces", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "affinity.traces means")
writeData(wb, sheet = "affinity.traces means", posthoc$emmeans)
addWorksheet(wb, "affinity.traces contrasts")
writeData(wb, sheet = "affinity.traces contrasts", posthoc$contrasts)

##################### pca.traces #####################

model <- lmer(pca.traces ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model pca.traces")
writeData(wb, sheet = "model pca.traces", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "pca.traces means")
writeData(wb, sheet = "pca.traces means", posthoc$emmeans)
addWorksheet(wb, "pca.traces contrasts")
writeData(wb, sheet = "pca.traces contrasts", posthoc$contrasts)

##################### tsne.traces #####################

model <- lmer(tsne.traces ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model tsne.traces")
writeData(wb, sheet = "model tsne.traces", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "tsne.traces means")
writeData(wb, sheet = "tsne.traces means", posthoc$emmeans)
addWorksheet(wb, "tsne.traces contrasts")
writeData(wb, sheet = "tsne.traces contrasts", posthoc$contrasts)

##################### affinity.spikes #####################

model <- lmer(affinity.spikes ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model affinity.spikes")
writeData(wb, sheet = "model affinity.spikes", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "affinity.spikes means")
writeData(wb, sheet = "affinity.spikes means", posthoc$emmeans)
addWorksheet(wb, "affinity.spikes contrasts")
writeData(wb, sheet = "affinity.spikes contrasts", posthoc$contrasts)

##################### tsne.spikes #####################

model <- lmer(tsne.spikes ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model tsne.spikes")
writeData(wb, sheet = "model tsne.spikes", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "tsne.spikes means")
writeData(wb, sheet = "tsne.spikes means", posthoc$emmeans)
addWorksheet(wb, "tsne.spikes contrasts")
writeData(wb, sheet = "tsne.spikes contrasts", posthoc$contrasts)

##################### LOAD DATA FOR EXPLAINED VAR #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/figure 6/slope_var.xlsx")
condition <- factor(data$condition)
mouse <- factor(data$mouse)
slope.var <- data$slope.var

##################### explained variance #####################

model <- lmer(slope.var ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model slope.var")
writeData(wb, sheet = "model slope.var", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "slope.var means")
writeData(wb, sheet = "slope.var means", posthoc$emmeans)
addWorksheet(wb, "slope.var contrasts")
writeData(wb, sheet = "slope.var contrasts", posthoc$contrasts)

##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
