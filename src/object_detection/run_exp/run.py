from object_detection.run_exp.mod import main as train
from object_detection.run_exp.pred import main as predict
from object_detection.run_exp.postproc import main as postprocess
from object_detection.run_exp.eval import main as evaluate
import argparse
import csv
import yaml

def main(experiment):
    # Define the paths
    csv_path = f'experiments/models/{experiment}/experiment_config.csv'
    yaml_template_path = 'experiments/config_template.yaml'
    output_dir = 'src/object_detection/config/experiment_config/'
    print(f"Output directory: {output_dir}")
    print(f"CSV path: {csv_path}")
    print(f"YAML template path: {yaml_template_path}")

    # Read the CSV file
    with open(csv_path, mode='r') as csv_file:
        csv_reader = list(csv.DictReader(csv_file))
    for row in csv_reader:
        train(row, yaml_template_path, output_dir, experiment, csv_path, csv_reader)

    predict(experiment)
    postprocess(experiment)
    evaluate(experiment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp','--experiment', type=str, required=True)
    args = parser.parse_args()
    main(args.experiment)