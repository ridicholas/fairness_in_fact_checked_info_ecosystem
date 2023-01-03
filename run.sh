cd 4_simulation/src
nohup python3 -u main.py -p 'pre' > pre.out &

wait

nohup python3 -u main.py -p 'post' -m 'None' > post_none.out & 

nohup python3 -u main.py -p 'post' -m 'stop_reading_misinfo' -l 'random' -s 'nodes_visited' > post_random_visited.out & 

nohup python3 -u main.py -p 'post' -m 'stop_reading_misinfo' -l 'stratified' -s 'nodes_visited' > post_strat_visited.out & 

nohup python3 -u main.py -p 'post' -m 'stop_reading_misinfo' -l 'knowledgable_community' -s 'nodes_visited' > post_know_visited.out & 

nohup python3 -u main.py -p 'post' -m 'stop_reading_misinfo' -l 'random' -s 'stratified_nodes_visited' > post_random_strat.out & 

nohup python3 -u main.py -p 'post' -m 'stop_reading_misinfo' -l 'stratified' -s 'stratified_nodes_visited' > post_strat_strat.out & 

nohup python3 -u main.py -p 'post' -m 'stop_reading_misinfo' -l 'knowledgable_community' -s 'stratified_nodes_visited' > post_know_strat.out & 


'''
cd ../../5_analysis/src
python3 process_data.py
python3 make_plots.py
'''
