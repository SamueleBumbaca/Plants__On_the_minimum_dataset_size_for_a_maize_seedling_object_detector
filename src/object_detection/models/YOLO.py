import ultralytics
from ultralytics import YOLO
import torch
from os.path import join

class YOLO_model(YOLO):
    def __init__(self,cfg):
        super(YOLO_model, self).__init__(cfg['train']['model']+'.pt')
        ultralytics.checks()
        self.cfg = cfg
        self.yolo_data = join(cfg['data']['yolo_dataset'], 'dataset.yaml')
    def train_model(self):
        self.train(data=self.yolo_data, 
                    epochs=self.cfg['train']['max_epoch'], 
                    imgsz=self.cfg['dataset']['image_size'],
                    batch=self.cfg['train']['batch_size'],
                    cache='disk',
                    device=  0 if torch.cuda.is_available() else 'cpu',
                    workers = self.cfg['train']['n_workers'],
                    project = join('experiments','models',self.cfg['experiment']['note']),
                    name = self.cfg['experiment']['id'],
                    pretrained = self.cfg['train']['use_pretrained'],
                    optimizer = self.cfg['train']['optimizer'],
                    seed = self.cfg['train']['seed'],
                    patience = self.cfg['train']['patience'],
                    deterministic = True,
                    single_cls = True,
                    multi_scale = False,
                    plots = False,
                    )
