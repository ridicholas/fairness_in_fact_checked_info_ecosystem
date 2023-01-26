library(data.table)
library(fixest)

### Overall Efficacy

results <- fread('~/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/5_analysis/output/individual_level_regression_data.csv',
                 colClasses = c('factor', 'factor', 'factor', 'factor', 'factor', 'numeric', 'numeric','numeric', 'numeric', 'numeric','numeric'))
results <- results[Community != 1,]

results[,Community := factor(Community,
                             levels = c('3', '56', '43', '72', '34', '127', '12', '28'))]

results[,Intervention:= factor(Intervention, 
                              levels = c('no_intervention_change_in_belief', 
                              'TopPredictedByTopic_knowledgable_community_stratified_nodes_visited',
                              'TopPredictedByTopic_knowledgable_community_nodes_visited',
                              'TopPredictedByTopic_random_nodes_visited',
                              'TopPredictedByTopic_random_stratified_nodes_visited',
                              'TopPredictedByTopic_stratified_nodes_visited',
                              'TopPredictedByTopic_stratified_stratified_nodes_visited',
                              'TopPredicted_knowledgable_community_nodes_visited',
                              'TopPredicted_knowledgable_community_stratified_nodes_visited',
                              'TopPredicted_random_nodes_visited',
                              'TopPredicted_random_stratified_nodes_visited',
                              'TopPredicted_stratified_nodes_visited',
                              'TopPredicted_stratified_stratified_nodes_visited'))]



colnames(results) <- c('Rep', 'Node', 'Topic', 'Intervention', 'Community', 'Number.of.Bots.Followed', 'Betweenness.Centrality', 'Impactedness', 'Average.External.Belief', 'Midpoint.Belief', 'Change.in.Belief')



mod <- feols(Change.in.Belief ~ 
               Rep + Rep:Community + 
               Intervention + Intervention:Community +
               Midpoint.Belief + Midpoint.Belief^2 + 
               Average.External.Belief + Average.External.Belief^2 + 
               Betweenness.Centrality + Betweenness.Centrality^2 + 
               Number.of.Bots.Followed + Number.of.Bots.Followed^2 + 
               Impactedness + Impactedness^2,
               data = results)
summary(mod)

coeftable(mod, cluster = ~Community + Rep)

### V2 - Compared to Baseline Model

results <- fread('~/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/5_analysis/output/individual_level_regression_data.csv',
                 colClasses = c('factor', 'factor', 'factor', 'factor', 'factor', 'numeric', 'numeric','numeric', 'numeric', 'numeric','numeric'))
results <- results[Community != 1 & Intervention != 'no_intervention_change_in_belief',]

results[,Community := factor(Community,
                             levels = c('3', '56', '43', '72', '34', '127', '12', '28'))]

results[,Intervention:= factor(Intervention, 
                               levels = c('TopPredicted_random_nodes_visited',
                                          'TopPredictedByTopic_knowledgable_community_stratified_nodes_visited',
                                          'TopPredictedByTopic_knowledgable_community_nodes_visited',
                                          'TopPredictedByTopic_random_nodes_visited',
                                          'TopPredictedByTopic_random_stratified_nodes_visited',
                                          'TopPredictedByTopic_stratified_nodes_visited',
                                          'TopPredictedByTopic_stratified_stratified_nodes_visited',
                                          'TopPredicted_knowledgable_community_nodes_visited',
                                          'TopPredicted_knowledgable_community_stratified_nodes_visited',
                                          'TopPredicted_random_stratified_nodes_visited',
                                          'TopPredicted_stratified_nodes_visited',
                                          'TopPredicted_stratified_stratified_nodes_visited'))]



colnames(results) <- c('Rep', 'Node', 'Topic', 'Intervention', 'Community', 'Number.of.Bots.Followed', 'Betweenness.Centrality', 'Impactedness', 'Average.External.Belief', 'Midpoint.Belief', 'Change.in.Belief')



mod <- feols(Change.in.Belief ~ 
               Rep + Rep:Community + 
               Intervention + Intervention:Community +
               Midpoint.Belief + Midpoint.Belief^2 + 
               Average.External.Belief + Average.External.Belief^2 + 
               Betweenness.Centrality + Betweenness.Centrality^2 + 
               Number.of.Bots.Followed + Number.of.Bots.Followed^2 + 
               Impactedness + Impactedness^2,
             data = results)
summary(mod)
coeftable(mod, cluster = ~Community + Rep)











