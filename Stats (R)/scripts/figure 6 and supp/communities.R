
Sys.setenv(LANG = "en")
rm(list  =  ls())

library('lme4')
library('lmerTest')
library('emmeans')
library('openxlsx')

##################### LOAD DATA FOR COMMUNITIES NO GAMMA #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/figure 6/communities1.xlsx")
condition <- factor(data$condition)
mouse <- factor(data$mouse)
MRCCnum_comm <- data$MRCC_num_comm
distance <- data$ED

excel_file <- "E:/Calcium Imaging/stats/stats summary/figure6 communities.xlsx"
wb <- createWorkbook()

##################### MRCCnum_comm #####################

model <- lmer(MRCCnum_comm ~ condition + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model MRCCnum_comm")
writeData(wb, sheet = "model MRCCnum_comm", anova(model))

posthoc <- emmeans(model, pairwise ~ condition)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "MRCCnum_comm means")
writeData(wb, sheet = "MRCCnum_comm means", posthoc$emmeans)
addWorksheet(wb, "MRCCnum_comm contrasts")
writeData(wb, sheet = "MRCCnum_comm contrasts", posthoc$contrasts)

##################### LOAD DATA FOR COMMUNITIES OVER GAMMA #####################

data <- read.xlsx("E:/Calcium Imaging/stats/datasets/figure 6/communities30.xlsx")
condition <- factor(data$condition)
mouse <- factor(data$mouse)
gamma <- factor(data$gamma)
num_comm_over_gamma <- data$num_comm_no_singl
modularity <- data$modularity
max_size <- data$max_size

##################### num_comm_over_gamma #####################

model <- lmer(num_comm_over_gamma ~ condition * gamma + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model num_comm_over_gamma")
writeData(wb, sheet = "model num_comm_over_gamma", anova(model))

posthoc <- emmeans(model, pairwise ~ condition | gamma)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "num_comm_over_gamma means")
writeData(wb, sheet = "num_comm_over_gamma means", posthoc$emmeans)
addWorksheet(wb, "num_comm_over_gamma contrasts")
writeData(wb, sheet = "num_comm_over_gamma contrasts", posthoc$contrasts)

##################### modularity #####################

model <- lmer(modularity ~ condition * gamma + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model modularity")
writeData(wb, sheet = "model modularity", anova(model))

posthoc <- emmeans(model, pairwise ~ condition | gamma)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "modularity means")
writeData(wb, sheet = "modularity means", posthoc$emmeans)
addWorksheet(wb, "modularity contrasts")
writeData(wb, sheet = "modularity contrasts", posthoc$contrasts)

##################### max size over gamma #####################

model <- lmer(max_size ~ condition * gamma + (1 | mouse)) # fit the model
anova(model) # test it using lmerTest way (ANOVA with Satterthwaite's method)
addWorksheet(wb, "model max_size")
writeData(wb, sheet = "model max_size", anova(model))

posthoc <- emmeans(model, pairwise ~ condition | gamma)
summary(posthoc) # print the post-hoc results
addWorksheet(wb, "max_size means")
writeData(wb, sheet = "max_size means", posthoc$emmeans)
addWorksheet(wb, "max_size contrasts")
writeData(wb, sheet = "max_size contrasts", posthoc$contrasts)


##################### SAVE EVERYTHING TO EXCEL #####################

saveWorkbook(wb = wb, excel_file)
