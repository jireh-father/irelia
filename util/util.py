from optparse import OptionParser
from util import constant


def parse_args():
    parser = OptionParser()

    parser.add_option("-a", "--first_agent_q_path", dest="first_agent_q_path", default="first_agent_q")
    parser.add_option("-b", "--second_agent_q_path", dest="second_agent_q_path", default="second_agent_q")
    # from_scratch or fine_tune
    parser.add_option("-c", "--train_mode", dest="train_mode", default=constant.ARG_TRAIN_MODE_FINE_TUNE)

    parser.add_option("-d", "--first_agent_name", dest="first_agent_name", default="first_agent")
    parser.add_option("-e", "--second_agent_name", dest="second_agent_name", default="second_agent")

    parser.add_option("-f", "--discount_reward_factor", dest="discount_reward_factor", default=0.9)
    parser.add_option("-g", "--num_episodes", dest="num_episodes", default=1000)

    # parser.add_option("-b", "--dataset_path", dest="dest", default="/home/temp/main_image_prediction")
    # parser.add_option("-n", "--dataset_name", dest="dataset_name", default="main_image")
    # parser.add_option("-m", "--model_name", dest="model_name", default="alexnet_v2")
    # parser.add_option("-p", "--preprocessing_name", dest="preprocessing_name", default="apparel")
    # parser.add_option("-t", "--num_preprocessing_threads", dest="num_preprocessing_threads", default=1)
    # parser.add_option("-c", "--checkpoint_path", dest="checkpoint_path", default="/home/model/main_image")
    # parser.add_option("-l", "--log_dir", dest="log_dir", default="/home/temp/main_image_prediction/log")
    # parser.add_option("-e", "--exe_mode", dest="exe_mode", default='worker')
    # parser.add_option("-r", "--params", dest="params",
    #                   default='{"mall_id":"test","prd_no":1,"mode":"sync","img_url_list":["http:\/\/img.hovits.com\/testimg\/1.jpg","http:\/\/img.hovits.com\/testimg\/2.jpg","http:\/\/img.hovits.com\/testimg\/a.jpg"]}')
    # parser.add_option("-v", "--log_mode", dest="log_mode", default="remove")

    options, _ = parser.parse_args()
    options.discount_reward_factor = float(options.discount_reward_factor)
    options.num_episodes = int(options.num_episodes)
    return options
