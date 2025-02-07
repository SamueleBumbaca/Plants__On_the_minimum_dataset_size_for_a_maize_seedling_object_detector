from object_detection.utils.predict import main as predict
from torch import cuda
import argparse
import csv
from os import makedirs

def main(experiment):
    # Define the paths
    csv_path = f'experiments/{experiment}/experiment_config.csv'
    output_dir = 'src/object_detection/config/experiment_config/'
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    # Read the CSV file
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if bool(row['done']):
                print(f"Predicting {experiment} experiment: {row['experiment']} with model: {row['model']} on dataset: {row['dataset']}")
                exp_id = f"{row['model']}_dataset_{row['dataset']}_exp_{experiment}_{row['experiment']}"
                cfg = output_dir + f"config_{exp_id}.yaml"
                ckpt = f'experiments/{experiment}/{exp_id}/weights/best.pt'
                out_folder = f'experiments/{experiment}/{exp_id}/predictions'
                # Run the experiments        
                predict(cfg, ckpt, out_folder)
                # # Update the CSV file
                # with open(csv_path, mode='r') as csv_file:
                #     csv_reader = csv.DictReader(csv_file)
                #     for row in csv_reader:
                #         if row['model'] == row['model'] and row['dataset'] == row['dataset']:
                #             row['done'] = True
                #             break
                #     # Write the new row
                #     with open(csv_path, mode='w') as csv_file:
                #         csv_writer = csv.DictWriter(csv_file, fieldnames=csv_reader.fieldnames)
                #         csv_writer.writeheader()
                #         for row in csv_reader:
                #             csv_writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiments')
    parser.add_argument('experiment', type=str, help='The experiment to run')
    args = parser.parse_args()
    main(args.experiment)