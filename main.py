import os
import pandas as pd
from PIL import Image
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection import utils, engine
from detection import transforms as T


def get_transform():
    
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    transforms.append(T.RandomHorizontalFlip(0.5))   
    return T.Compose(transforms) 

def get_model(num_classes):    
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
   
    return model



def parse_one_annot(path_to_data_file, filename):
   data = pd.read_csv(path_to_data_file)
   boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
   return boxes_array


class LPRDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(root))
        self.path_to_data_file = data_file
        
    def __getitem__(self, idx):    
        # load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_annot(self.path_to_data_file, self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    def __len__(self):
             return len(self.imgs)
     

if __name__ == '__main__':
    ## 1. Config
    img_train_path = './dataset/train'
    img_val_path = './dataset/val'
    train_csv = './dataset/train.csv'
    val_csv = './dataset/val.csv'
    
    
    ## 2. Set Dataset
    dataset_train = LPRDataset(img_train_path, train_csv,
                               transforms=get_transform())
    
    dataset_val = LPRDataset(img_val_path, val_csv,
                             transforms=get_transform())
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                              batch_size=1, 
                                              shuffle=True, 
                                              num_workers=4, 
                                              collate_fn=utils.collate_fn)
    
    data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                              batch_size=1, 
                                              shuffle=False, 
                                              num_workers=4, 
                                              collate_fn=utils.collate_fn)
    
    ## 3. Prepare for Training
    print(torch.cuda.is_available())
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2 #lpr n not lpr
    model = get_model(num_classes)
    model.to(device)
    
    # 4. Hyper Param
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        engine.evaluate(model, data_loader_val, device=device)
    
    
    
    
    
    