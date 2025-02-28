from object_detection.train_indomain import main as train
from object_detection.train_outdomain import main as train_with_suplemental_data
from torch import cuda
import argparse
import csv
import yaml
import gc
from os.path import join, exists

def main(row, yaml_template_path, output_dir, experiment, csv_path, csv_reader):
    # Define the paths   
    if row['done']:
        print(f"Skipping {experiment} experiment: {row['experiment']} with model: {row['model']} on dataset: {row['dataset']}")
        print(f"Experiment already done")
    else:
        print(f"Running {experiment} experiment: {row['experiment']} with model: {row['model']} on dataset: {row['dataset']}")
        # Read the YAML template
        with open(yaml_template_path, 'r') as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
        
        # Substitute the placeholders in the YAML content
        yaml_content['experiment']['id'] = f"{row['model']}_dataset_{row['dataset']}_exp_{experiment}_{row['experiment']}"
        yaml_content['config_path'] = output_dir + f"config_{yaml_content['experiment']['id']}.yaml"
        yaml_content['data']['dataset'] = row['dataset']
        yaml_content['data']['train_split'] = row['train_size']
        yaml_content['data']['val_split'] = row['val_size']
        yaml_content['data']['dataset_size'] = row['dataset_size'] if row['dataset_size'] else 'all'
        yaml_content['train']['model'] = row['model']
        yaml_content['experiment']['note'] = experiment
        if row['pre-trained'] == 'custom':
            yaml_content['train']['backbone'] = row['backbone']
        elif row['pre-trained'] == 'default':
            yaml_content['train']['backbone'] = 'default'
        out_domain = row['out_domain']

        # If the dataset size is not specified, use all the data
        if row['dataset_size']:
            dataset_size = int(row['dataset_size'])
            print(f'Using the training data size {dataset_size}')
            tsp = int(float(row['train_size'])*dataset_size)
            vsp = int(float(row['val_size'])*dataset_size)
            dtq = int(float(row['dataset_quality'])*100)
        else:
            print('Using all the training data')
            dataset_size = 'all'
            tsp = int(float(row['train_size'])*100)
            vsp = int(float(row['val_size'])*100)
            dtq = int(float(row['dataset_quality'])*100)

        # Create the YOLO dataset
        yolo_dataset_folder = f'DatasetSize_{dataset_size}_Train_{tsp}_Val_{vsp}_Quality_{dtq}_Dataset_{yaml_content['data']['dataset']}'
        if out_domain == 'no':
            yolo_dataset_path = join(yaml_content['data']['path_to_dataset'], 
                                    yaml_content['data']['dataset'], 
                                    'yolo_datasets' , 
                                    yolo_dataset_folder)
        if out_domain == 'yes':
            yolo_dataset_path = join(yaml_content['data']['path_to_dataset'], 
                                    'out_domain', 
                                    'yolo_datasets',
                                    yaml_content['data']['dataset'],
                                    yolo_dataset_folder)
        
        yaml_content['data']['yolo_dataset'] = yolo_dataset_path

        # Set the dataset quality
        if row['dataset_quality']:
            yaml_content['data']['dataset_quality'] = float(row['dataset_quality'])
        
        # Write the new YAML file
        with open(yaml_content['config_path'], 'w') as new_yaml_file:
            yaml.dump(yaml_content, new_yaml_file)

        # clean memory and CUDA chache to avoid memory issues
        cuda.empty_cache()
        gc.collect()

        # Run the experiments
        try:        
            if out_domain == 'yes':
                print('Training with out of domain suplemental data')
                train_with_suplemental_data(yaml_content['config_path'])
            else:
                print('Training with the in domain data')
                train(yaml_content["config_path"])

            # Update the config file with the model checkpoint path
            ckpt_best_path = f"experiments/models/{experiment}/{yaml_content['experiment']['id']}/weights/best.pt"
            ckpt_last_path = f"experiments/models/{experiment}/{yaml_content['experiment']['id']}/weights/last.pt"
            if exists(ckpt_best_path):
                yaml_content['train']['model_checkpoint'] = ckpt_best_path
                ckpt_path = ckpt_best_path
            elif exists(ckpt_last_path):
                yaml_content['train']['model_checkpoint'] = ckpt_last_path
                ckpt_path = ckpt_last_path
            else:
                raise AssertionError(f"Model checkpoint not found in {ckpt_best_path} or {ckpt_last_path}")
            # Update the config file with the model checkpoint path
            yaml_content['train']['model_checkpoint'] = ckpt_path
            with open(yaml_content['config_path'], 'w') as new_yaml_file:
                yaml.dump(yaml_content, new_yaml_file)

            # Update the CSV file
            row['done'] = '_' # Mark the experiment as done

            print(f"Experiment {yaml_content['experiment']['id']} finished successfully")
    
        except Exception as e:
            if e == AssertionError:
                print(f"Assertion error: {e}")
                # Update the CSV file with the error
                row['done'] = 'x' # Mark the experiment as undone
            else:
                print("Caught an exception, continuing with the next model or finising the experiment")
                print(f"Exception: {e}")

        # Write the updated row back to the CSV file
        finally:
            with open(csv_path, mode='w') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=csv_reader[0].keys())
                csv_writer.writeheader()
                csv_writer.writerows(csv_reader)

            # Clean up memory
            del yaml_content
            cuda.empty_cache()
            gc.collect()
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment with the given configuration.')
    parser.add_argument('-exp', '--experiment', type=str, required=True, help='Name of the experiment')
    args = parser.parse_args()
    main(args.experiment)