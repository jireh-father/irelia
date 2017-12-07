# irelia
under construction...




# Prerequisite
- pip install colorama
- ...





# self-play and train with no dataset
```shell
python self_play_and_train.py --save_dir="your path to save your model and self-play dataset" --max_step=100 --max_episode=10000 --max_simulation=200 --episode_interval_to_train=10 --print_mcts_tree=False --print_mcts_search=False
```





# train with dataset

## download the dataset
- 38MB
- [Download the dataset](http://img.hovits.com/korean-chess-records-dataset.txt)
```shell
mkdir dataset
cd dataset
wget http://img.hovits.com/korean-chess-records-dataset.txt
```


## convert the dataset to the real dataset for training
```shell
python parse_dataset.py --dataset_dir="the path you downloaded"
```


## train with dataset
```shell
python optimizer.py --dataset_dir="the path you converted" --save_dir="your path to save your model" --epoch=10 --num_model_layers=20 --batch_size=32
```





# Play with trained AI with MCTS
```shell
python user_vs_trained_mcts.py --save_dir="the model dir you trained" --model_file_name="the model name you trained" --max_step=100 --max_episode=10000 --max_simulation=200 --print_mcts_tree=False --print_mcts_search=False
```





# Trained AI's self-play with MCTS
```shell
python play_trained_mcts_vs_trained_mcts.py --save_dir="the model dir you trained" --model_file_name="the model name you trained" --max_step=100 --max_episode=10000 --max_simulation=200 --print_mcts_tree=False --print_mcts_search=False
```





# Trained AI's self-play with no MCTS
```shell
python play_net_vs_net.py --save_dir="the model dir you trained" --model_file_name="the model name you trained" --max_step=100 --max_episode=10000 --max_simulation=200 --print_mcts_tree=False --print_mcts_search=False
```





# review site
It is a review site that recorded a match between learned AIs with dataset, and it is the first version that learns using Q-table learning using dynamic state.


it is not an algorithm of alpha high zero.


http://irelia.hovits.com/web/review.html