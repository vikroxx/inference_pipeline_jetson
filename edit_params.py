import json
import argparse


def app(param_json, new_detect_source, new_detect_dest, weight_file_name):
    params = json.load(open(param_json, 'r'))
    if weight_file_name is not None:
        params['detect']['weights_file_name'] = weight_file_name
    params['detect']['source'] = new_detect_source
    params['detect']['output'] = new_detect_dest
    json.dump(params, open(param_json, 'w'), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--json', type=str)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    app(args.json, args.source, args.output, args.weights)
