import cv2
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection import transforms as T


def get_model(num_classes):    
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
   
    return model

if __name__ == '__main__':
    path_output = "./model.pth"
    
    loaded_model = get_model(num_classes = 2)
    loaded_model.load_state_dict(torch.load(path_output))
    loaded_model.eval()
    
    vidSrc = 'vid1.mp4'
    saveVid = True

    cap = cv2.VideoCapture(vidSrc)
    if saveVid:
        vid_w = 640
        vid_h = 480
        savedName = "test.avi"
        vid = cv2.VideoWriter_fourcc('M','J','P','G')
        vout = cv2.VideoWriter(savedName, vid, 20.0, (vid_w, vid_h))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ## --- main program
            
            x = torch.from_numpy(frame).cuda()
            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
            with torch.no_grad():
                prediction = loaded_model([x])
            
            for element in range(len(prediction[0]["boxes"])):
                 bbox = prediction[0]["boxes"][element].cpu().numpy()
                 score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals= 3)
                 if score > 0.8:
                     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                     cv2.putText(frame, str(score),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 100, (0,255,0),2)
            
            
            ## ---
            if saveVid:
                frame_saved  = cv2.resize(frame, (vid_w, vid_h))
                vout.write(frame_saved)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    if saveVid:
        vout.release()
    cv2.destroyAllWindows()


    
    
    
    
    