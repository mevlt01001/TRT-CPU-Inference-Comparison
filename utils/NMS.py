import torch
import torchvision

class out_splitter(torch.nn.Module):
    def __init__(self, score_threshold: float=0.5):
        super(out_splitter, self).__init__()
        self.score_threshold = score_threshold

    def forward(self, yolo_raw_output):
        out = torch.permute(yolo_raw_output, (0, 2, 1))[0] # (1,84,8400) to (8400,84)
        out = out[out[:, 4] > self.score_threshold]
        cxcywh = out[:, :4] # (8400,4)
        person_conf = out[:, 4]
        return cxcywh, person_conf

class cxcywh_to_xyxy(torch.nn.Module):
    def __init__(self):
        super(cxcywh_to_xyxy, self).__init__()

    def forward(self, cxcywh):
        xyxy = torchvision.ops.box_convert(cxcywh, "cxcywh", "xyxy")
        return xyxy
    
class NMS(torch.nn.Module):
    def __init__(self, iou_threshold: float=0.5):
        super(NMS, self).__init__()
        self.iou_threshold = iou_threshold

    def forward(self, xyxy, person_conf):
        selected_indices = torchvision.ops.nms(xyxy, person_conf, self.iou_threshold)
        boxes_and_scores = torch.cat((xyxy, person_conf.unsqueeze(1)), dim=1)
        return boxes_and_scores[selected_indices]
    
class YOLO12_postprocess(torch.nn.Module):
    def __init__(self, score_threshold: float=0.5, iou_threshold: float=0.5):
        super(YOLO12_postprocess, self).__init__()
        self.out_splitter = out_splitter(score_threshold)
        self.cxcywh_to_xyxy = cxcywh_to_xyxy()
        self.NMS = NMS(iou_threshold)

    def forward(self, yolo_raw_output):
        cxcywh, person_conf = self.out_splitter(yolo_raw_output)
        xyxy = self.cxcywh_to_xyxy(cxcywh)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores