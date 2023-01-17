cd 4_simulation/src
nohup python3 -u main.py -p 'pre' > pre.out &

wait

nohup python3 -u main.py -p 'post' -m 'None' > post_none.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredicted' -l 'random' -s 'nodes_visited' > post_random_visited.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredicted' -l 'stratified' -s 'nodes_visited' > post_strat_visited.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredicted' -l 'knowledgable_community' -s 'nodes_visited' > post_know_visited.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredicted' -l 'random' -s 'stratified_nodes_visited' > post_random_strat.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredicted' -l 'stratified' -s 'stratified_nodes_visited' > post_strat_strat.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredicted' -l 'knowledgable_community' -s 'stratified_nodes_visited' > post_know_strat.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredictedByTopic' -l 'random' -s 'nodes_visited' > post_random_visited.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredictedByTopic' -l 'stratified' -s 'nodes_visited' > post_strat_visited.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredictedByTopic' -l 'knowledgable_community' -s 'nodes_visited' > post_know_visited.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredictedByTopic' -l 'random' -s 'stratified_nodes_visited' > post_random_strat.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredictedByTopic' -l 'stratified' -s 'stratified_nodes_visited' > post_strat_strat.out & 

nohup python3 -u main.py -p 'post' -m 'TopPredictedByTopic' -l 'knowledgable_community' -s 'stratified_nodes_visited' > post_know_strat.out & 



