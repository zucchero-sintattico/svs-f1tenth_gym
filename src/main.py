import argparse
import os
import random

from train import train, run


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
map_path = os.path.join(base_path, "map", "f1tenth_racetracks")

def get_available_maps():
    dir_content = os.listdir(map_path)
    maps = []
    for item in dir_content:
        if os.path.isdir(os.path.join(map_path, item)):
            maps.append(item)
    return maps


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

    run_parser = subparser.add_parser('run', help='Run the model in the simulator')
    run_parser.add_argument('-m', '--map', type=str, help='Map to train the model on. Default: random maps', default='random')
    run_parser.add_argument('--timesteps', type=float, help='Number of timesteps to train the model. Default: 0.01', default=0.01)

    args = parser.parse_args()

    available_maps = get_available_maps()
    if 'random' not in args.map and args.map not in available_maps:
        print(f"Map '{args.map}' not found. Available maps:")
        print(available_maps)
        exit(1)

    if args.list:
        print("Available maps:")
        print(available_maps)
        exit(0)
    elif args.command == 'train':
        print("Training the model")
        train(args.map == 'random',
                args.map,
                args.timesteps,
                args.min_timesteps,
                args.max_timesteps,
                args.min_learning_rate,
                args.max_learning_rate,
                args.num_of_steps)
        exit(0)
    elif args.command == 'run':
        print("Running the model")
        if args.map == 'random':
            args.map = random.choice(available_maps)
        run(args.map, args.timesteps)
        exit(0)
    else:
        print("Invalid command")
        exit(1)