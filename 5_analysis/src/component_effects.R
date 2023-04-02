library(data.table)
library(fixest)
library(ggplot2)
library(dplyr)
### Overall Efficacy

communities <- c(3, 34)
workflow_levels <- c('TopPredicted', 'TopPredictedByTopic', 'None')
labeling_levels <- c('Random Labeling', 'Stratified Labeling', 'Knowledgeable Community Labeling', 'None')
sampling_levels <- c('Virality Sampling', 'Stratified Virality Sampling', 'None')
reps = 10



results <- fread('/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/5_analysis/output/world_state_2/individual_level_regression_data.csv',
                 colClasses = c('factor', 'factor', 'factor', 'factor', 'factor', 'numeric','numeric','numeric', 'numeric','numeric', 'numeric', 'numeric','numeric'))
colnames(results) <- c('Rep', 'Node', 'Topic', 'Intervention', 'Community', 'Number.of.Bots.Followed', 'Betweenness.Centrality','Clustering','Number.Followed.In.Other.Comms', 'Impactedness', 'Average.External.Belief', 'Midpoint.Belief', 'Change.in.Belief')

iwcib <- results %>%
  group_by(Rep, Node) %>%
  mutate(ImpactednessPct = Impactedness/sum(Impactedness)) %>%
  ungroup() %>%
  group_by(Rep, Node, Intervention, Community) %>%
  summarise(iwcib = sum(Change.in.Belief*ImpactednessPct)) %>%
  setDT()

null <- iwcib %>%
  filter(grepl('no_intervention', Intervention)) %>%
  select(Node, Rep, iwcib) %>%
  rename(iwcib_null = iwcib)


treatment_effect <- iwcib %>%
  filter(!grepl('no_intervention', Intervention)) %>%
  left_join(null) %>%
  mutate(te = iwcib - iwcib_null) %>%
  setDT()

treatment_effect[,InterventionWorkflow := factor(ifelse(grepl("TopPredictedByTopic", Intervention), 'TopPredictedByTopic',
                                               ifelse(grepl('TopPredicted', Intervention), 'TopPredicted', 'None')), levels = workflow_levels)]

treatment_effect[,InterventionLabeling := factor(ifelse(grepl("no_intervention", Intervention), 'None',
                                               ifelse((grepl('TopPredicted_stratified_', Intervention) | grepl('TopPredictedByTopic_stratified_', Intervention)), 'Stratified Labeling',
                                                      ifelse(grepl('knowledgable_community', Intervention), 'Knowledgeable Community Labeling', 'Random Labeling'))), levels = labeling_levels)]

treatment_effect[,InterventionSampling := factor(ifelse(grepl('no_intervention', Intervention), 'None',
                                               ifelse(InterventionLabeling != 'Stratified Labeling' & grepl('stratified_nodes_visited', Intervention), 'Stratified Virality Sampling', 
                                                      ifelse(InterventionLabeling == 'Stratified Labeling' & grepl('stratified_stratified_nodes_visited', Intervention), 'Stratified Virality Sampling', 'Virality Sampling'))), levels = sampling_levels)]



component_effect_results <- data.frame()


for (i in 1:reps) {
  
  rep_results <- subset(treatment_effect, Rep == (i - 1))
  network_mod <- lm(te ~ InterventionWorkflow + InterventionLabeling + InterventionSampling, data = rep_results)
  coefs <- summary(network_mod)$coef
  
  row <- data.frame(rep = i, 
                    effect = as.character('Network'), 
                    intercept = coefs[1,1],
                    TopPredictedByTopic = coefs[2,1],
                    StratifiedLabeling = coefs[3,1],
                    KCLabeling = coefs[4,1],
                    StratifiedViralitySampling = coefs[5,1])
  
  component_effect_results <- rbind(component_effect_results, row)
  
  
  for (c in communities) {
    
    rep_results_com <- subset(rep_results, Community == c)
    network_com_mod <- lm(te ~ InterventionWorkflow + InterventionLabeling + InterventionSampling, data = rep_results_com)
    
    if (c == 3) {
      community_label <- as.character('Majority')
    } else {
      community_label <- as.character('Minority')
    }
    
    coefs <- summary(network_com_mod)$coef
    
    row <- data.frame(rep = i, 
                      effect = community_label, 
                      intercept = coefs[1,1],
                      TopPredictedByTopic = coefs[2,1],
                      StratifiedLabeling = coefs[3,1],
                      KCLabeling = coefs[4,1],
                      StratifiedViralitySampling = coefs[5,1])
    component_effect_results <- rbind(component_effect_results, row)
    
  }
  
}





component_effect_results <- component_effect_results %>%
  select(-rep, -intercept) %>%
  group_by(effect) %>%
  summarise_all(mean)

component_effect_results <- data.frame()


for (i in 1:reps) {
  
  rep_results <- subset(treatment_effect, Rep == (i - 1))
  network_mod <- lm(te ~ InterventionLabeling, data = rep_results)
  coefs <- summary(network_mod)$coef
  
  row <- data.frame(rep = i, 
                    effect = as.character('Network'), 
                    RandomLabel = coefs[1,1],
                    StratifiedLabeling = coefs[2,1],
                    KCLabeling = coefs[3,1])
  
  component_effect_results <- rbind(component_effect_results, row)
  
  
  for (c in communities) {
    
    rep_results_com <- subset(rep_results, Community == c)
    network_com_mod <- lm(te ~ InterventionLabeling + InterventionSampling, data = rep_results_com)
    
    if (c == 3) {
      community_label <- as.character('Majority')
    } else {
      community_label <- as.character('Minority')
    }
    
    coefs <- summary(network_com_mod)$coef
    
    row <- data.frame(rep = i, 
                      effect = community_label, 
                      RandomLabel = coefs[1,1],
                      StratifiedLabeling = coefs[2,1],
                      KCLabeling = coefs[3,1])
    
    component_effect_results <- rbind(component_effect_results, row)
    
  }
  
}




component_effect_results <- component_effect_results %>%
  select(-rep) %>%
  group_by(effect) %>%
  summarise_all(mean)




