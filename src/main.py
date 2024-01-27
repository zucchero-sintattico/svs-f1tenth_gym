import argparse
import os
import random

from train import train, run, evaluate


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
map_path = os.path.join(base_path, "map", "f1tenth_racetracks")

def get_available_maps():
    dir_content = os.listdir(map_path)
    maps = []
    for item in dir_content:
        if os.path.isdir(os.path.join(map_path, item)) and \
            os.path.exists(os.path.join(map_path, item, item + "_raceline.csv")):
            maps.append(item)
    return maps

def check_map(map_name):
    if map_name not in get_available_maps():
        print(f"Map '{map_name}' not found. Available maps:")
        print(get_available_maps())
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        This is the entry point for the application.
        You can train the model for autonomous driving or run the model in the simulator.""")
    parser.add_argument('-l', '--list', action='store_true', help='List all available maps')
    subparser = parser.add_subparsers(dest="command", help='Select the command to run')

    train_parser = subparser.add_parser('train', help='Train the model')
    train_parser.add_argument('-m', '--map', type=str, help='Map to train the model on. Default: random maps', default='random')
    train_parser.add_argument('--timesteps', type=float, help='Number of timesteps to train the model. Default: 0.01', default=0.01)
    train_parser.add_argument('--min-timesteps', type=int, help='Maximum number of timesteps to train the model. Default: 10_000', default=10_000)
    train_parser.add_argument('--max-timesteps', type=int, help='Maximum number of timesteps to train the model. Default: 11_000', default=11_000)
    train_parser.add_argument('--min-learning-rate', type=float, help='Learning rate for the model. Default: 0.0005', default=0.0005)
    train_parser.add_argument('--max-learning-rate', type=float, help='Learning rate for the model. Default: 0.0001', default=0.0001)
    train_parser.add_argument('--num-of-steps', type=int, help='Number of steps to train the model. Default: 10', default=10)
    train_parser.add_argument('--num-of-sub-steps', type=int, help='Number of steps for each traing cicle. Default: 10', default=10)
    train_parser.add_argument('--optimize-speed', action='store_true', help='Save the model to the specified path. Default: None')

    run_parser = subparser.add_parser('run', help='Run the model in the simulator')
    run_parser.add_argument('-m', '--map', type=str, help='Map to train the model on. Default: random maps', default='random')
    run_parser.add_argument('--timesteps', type=float, help='Number of timesteps to train the model. Default: 0.01', default=0.01)

    evaluate_parser = subparser.add_parser('evaluate', help='Evaluate the model')
    evaluate_parser.add_argument('-m', '--map', type=str, help='Map to train the model on. Default: random maps', default='random')
    evaluate_parser.add_argument('--timesteps', type=float, help='Number of timesteps to train the model. Default: 0.01', default=0.01)
    evaluate_parser.add_argument('--n-evaluate', type=int, help='Number of timesteps to train the model. Default: 20', default=20)

    args = parser.parse_args()

    available_maps = get_available_maps()

    if args.list:
        print("Available maps:")
        print(get_available_maps())
        exit(0)
    elif args.command == 'train':
        print("Training the model")
        if 'random' not in args.map:
            check_map(args.map)
        train(args.map == 'random',
                args.map,
                args.timesteps,
                args.min_timesteps,
                args.max_timesteps,
                args.min_learning_rate,
                args.max_learning_rate,
                args.num_of_steps,
                args.num_of_sub_steps,
                args.optimize_speed)
        exit(0)
    elif args.command == 'run':
        print("Running the model")
        if args.map == 'random':
            args.map = random.choice(available_maps)
        check_map(args.map)
        run(args.map, args.timesteps)
        exit(0)
    elif args.command == 'evaluate':
        print("Evaluating the model")
        if 'random' not in args.map:
            check_map(args.map)
        evaluate(
            args.map == 'random',
            random.choice(available_maps) if args.map == 'random' else args.map,
            args.timesteps,
            args.n_evaluate)
        exit(0)
    else:
        print("Invalid command")
        exit(1)
