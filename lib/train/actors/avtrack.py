from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F


class DGTrackActor(BaseActor):
    """ Actor for training DGTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """

        out_dict = self.forward_pass(data)


        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):

        if len(data['template_images']) == 3:
            for key in ['template_images', 'template_anno', 'template_masks', 'search_images', 'search_anno',
                        'search_masks', 'template_att', 'search_att']:
                if key in data:
                    if data[key].shape[0] == 3:
                        data[key] = torch.cat([data[key][0], data[key][1], data[key][2]], dim=0).unsqueeze(0)

        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        template_anno = data['template_anno']
        search_anno = data['search_anno']

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img, template_anno=template_anno, search_anno=search_anno)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):

        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        mine_loss = pred_dict['mine_loss']
        activeness_loss = pred_dict['activeness_loss']

        loss = (self.loss_weight['giou'] * giou_loss
                + self.loss_weight['l1'] * l1_loss
                + self.loss_weight['focal'] * location_loss
                + self.loss_weight['mine_loss'] * mine_loss
                + self.loss_weight['activeness_loss'] * activeness_loss)

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/mine_loss": mine_loss.item(),
                      "Loss/activeness_loss": activeness_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
