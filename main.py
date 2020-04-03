import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pascal_dataset import PASCALDataset
import utils
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from PennFudanDataset import PennFudanDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myDatasets = ['PASCAL', 'PennFudanPed_hw3']
learningRates = [0.01, 0.001, 0.0001]

for dataset in myDatasets:

    if dataset == 'PASCAL':
        class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane']
        dataset_train = PASCALDataset('PASCAL/train/')
        dataset_val = PASCALDataset('PASCAL/val/')

    else:
        class_names = ['background', 'person']
        dataset_train = PennFudanDataset('PennFudanPed_hw3/train/')
        dataset_val = PennFudanDataset('PennFudanPed_hw3/val/')
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_ftrs = model.roi_heads.box_predictor.bbox_pred.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, len(class_names))

    num_epochs = 5
    model = model.to(device)
    best_mAP = 0.0
    for lr in learningRates:

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_lr_mAP = 0.0
        dataset_sizes = {'train': len(dataset_train), 'val': len(dataset_val)}

        best_model_wts = copy.deepcopy(model.state_dict())
        

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    data_loader = data_loader_train
                else:
                    model.eval()
                    data_loader = data_loader_val
                    coco = get_coco_api_from_dataset(data_loader.dataset)
                    iou_types = ["bbox"]
                    coco_evaluator = CocoEvaluator(coco, iou_types)


                for images, targets in data_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            loss_dict = model(images, targets)
                            if isinstance(loss_dict, dict):
                                loss = loss_dict['loss_classifier'].add(loss_dict['loss_box_reg'])
                                loss = loss.add(loss_dict['loss_objectness'])
                                loss = loss.add(loss_dict['loss_rpn_box_reg'])
                                #loss = torch.stack([  loss_dict['loss_classifier'] , loss_dict['loss_box_reg'] , loss_dict['loss_objectness'] , loss_dict['loss_rpn_box_reg']  ], dim=0).sum(dim=0)
                                loss.backward()
                                optimizer.step()
                            #else:
                                #print(loss_dict)

                    if phase == 'val':
                        outputs = model(images)
                        #outputs = outputs.cpu()
                        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                        coco_evaluator.update(res)
                epoch_mAP = 0.0
                if phase == 'train':
                    scheduler.step()
                else:
                    coco_evaluator.synchronize_between_processes()
                    coco_evaluator.accumulate()
                    coco_evaluator.summarize()

                    epoch_mAP = coco_evaluator.coco_eval['bbox'].stats[0]

                if phase == 'val' and epoch_mAP > best_lr_mAP:
                    best_lr_mAP = epoch_mAP

                if phase == 'val' and epoch_mAP > best_mAP:
                    best_mAP = epoch_mAP
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts , 'best_model_weight_'+dataset+'.pth')

        print("Best mAP for "+dataset+" under Learning Rate "+str(lr)+": ", best_lr_mAP)
    
    print("Best mAP for "+dataset+": ", best_mAP)

