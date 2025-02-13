from object_detection.run_exp.mod import main as train
from object_detection.run_exp.pred import main as predict
from object_detection.run_exp.postproc import main as postprocess
from object_detection.run_exp.eval import main as evaluate
import argparse

def main(experiment):
    train(experiment)
    predict(experiment)
    postprocess(experiment)
    evaluate(experiment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp','--experiment', type=str, required=True)
    args = parser.parse_args()
    main(args.experiment)