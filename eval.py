import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('./')
sys.path.insert(0, '/home/jovyan/project/jbnu/yolov5')

from project.jbnu.yolov5.utils.logger import self_print as print, Logger
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
import cv2
import os.path as osp
from project.jbnu.preprocessing.util import draw_bboxes
from project.jbnu.preprocessing.preprocessing import preprocess_raw_data
from project.jbnu.yolov5.data.voc2yolo import voc2yolo_convert
from project.jbnu.yolov5.models.experimental import attempt_load
from project.jbnu.yolov5.utils.datasets import create_dataloader
from project.jbnu.yolov5.utils.general import check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from project.jbnu.yolov5.utils.loss import compute_loss
from project.jbnu.yolov5.utils.metrics import ap_per_class, ConfusionMatrix
from project.jbnu.yolov5.utils.plots import plot_study_txt
from project.jbnu.yolov5.utils.torch_utils import select_device, time_synchronized
from project.jbnu.yolov5.utils.json_utils import JsonTemplate


def draw_vis_img(path, predn, summary_path, result_summary, load_gt = False, filter_threshold = 0.5):

    selected_index =  predn[:, -2] >= filter_threshold
    selected_bbox = predn[selected_index]
    origin_img = cv2.imread(path.as_posix())
    pred_img = draw_bboxes(origin_img, {'bboxes': selected_bbox[:, :4].cpu().numpy(), 'name': selected_bbox[:,
                                                                                              4].cpu().numpy()},
                           color=(0, 255,
                                  0))
    if load_gt:
        vis_img = cv2.imread(path.as_posix().replace('img', 'vis'))
        pred_img = cv2.hconcat([vis_img, pred_img])

    name = str(path.name).split('.')[0]
    raw_name = name[:name.rfind('_')]
    cv2.imwrite(osp.join(summary_path.as_posix() , '%s.jpg'%raw_name), pred_img)

    result_summary.append({'img': raw_name, 'img_meta':'%s.%s'%(raw_name, 'json'),
                           'vis_img':'%s_vis.jpg'%raw_name})

def test(opt, data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         iou_overlap=0.5, # for mAP0.5
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=False,
         log_imgs=500, use_thread_pool = True):  # number of logged images

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        summary_path = save_dir
        summary_path.mkdir(exist_ok=True, parents=True)
        result_summary = []

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    data['val'] = osp.join(opt.input, 'val.txt')
    data['test'] = osp.join(opt.input, 'val.txt')
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # iouv = torch.linspace(iou_overlap, 1.0, 10).to(device)  # iou vector for mAP@0.5:0.95 #0.95 --> 1.0
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True, workers  = opt.workers)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@%.f'%iou_overlap,
                                 'mAP@%.1f:.95'%iou_overlap)
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images, json_templates = [], [], [], [], [], {}
    tp_fp_dict = defaultdict()

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()

            # list store  predict result [x1, y1, x2, y2, conf, cls]
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb)
            t1 += time_synchronized() - t

        # Statistics per image


        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            name = osp.split(path)[-1].split('.')[0]
            raw_name = name[:name.rfind('_')]
            # name --> class1 -- [tp, fp, gt], class2 -- [tp, fp, gt]]
            tp_fp_dict[raw_name] = {i:{'tp':[], 'fp':[], 'gt':[], 'fp_repeat':[]} for i in range(nl)}


            if not training:
                json_template = JsonTemplate(name = raw_name, raw_height= shapes[si][0][0], raw_width= shapes[si][0][0])
                json_template.set_raw_image_path(opt.raw_input, extension = opt.img_suffix)
                # TO /gt * width
                # gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                json_template.set_save_json_path(str(save_dir))
                json_template.set_save_img_path(str(save_dir))
                json_templates[raw_name] = json_template


            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))

                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    tp_fp_dict[raw_name][int(cls)]['gt'] = tbox[ti].cpu().numpy()

                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])



                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    tp_fp_dict[raw_name][int(cls)]['gt'] = tbox[ti].cpu().numpy()

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        pred_gt_matrix = box_iou(predn[pi, :4], tbox[ti])
                        ious, i = pred_gt_matrix.max(1)  # best ious, indices

                        # to label this bbox is select or not
                        is_select = np.zeros((len(predn[pi], )), dtype=np.bool)

                        # Append detections, tp, fp
                        detected_set = set()
                        # sorted from large to low iou and search match or not
                        overlap_index = (ious > iou_overlap).nonzero(as_tuple=False).flatten()
                        overlap_ious = ious[overlap_index]
                        sorted, indices = torch.sort(overlap_ious, descending=True)

                        for j in overlap_index[indices]:
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)

                                is_select[j] = True

                                correct[pi[j]] = ious[j] > iou_overlap#iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

                        for index in range(len(ious)):
                            *xyxy, conf, cls = predn[pi][index].cpu().numpy()
                            # not selected and iou large than iou_threshold
                            if not is_select[index]:
                                if ious[index] > iou_overlap:
                                    tp_fp_dict[raw_name][int(cls)]['fp_repeat'].append([conf, *xyxy])
                                else:
                                    tp_fp_dict[raw_name][int(cls)]['fp'].append([conf, *xyxy])
                            else:
                                tp_fp_dict[raw_name][int(cls)]['tp'].append([conf, *xyxy])

            # Append statistics (correct, conf, pcls, tcls) --> predict bbox, target bbox
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


    counter = defaultdict(int)
    for k, v in tp_fp_dict.items():
        counter['disease_tp'] += len(v[0]['tp'])
        counter['disease_fp'] += len(v[0]['fp'])
        counter['disease_fp_repeat'] += len(v[0]['fp_repeat'])
        counter['disease_gt'] += len(v[0]['gt'])

    # import pprint
    # pprint.pprint(counter, indent=4)


    if not training and opt.greed_search:
        best_f1, best_acc, best_map, best_threshold = -1, -1, -1, -1

        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
        best_map = map



        for threshold in tqdm(range(opt.greed_search_range[0], opt.greed_search_range[1], opt.greed_search_range[2]),
                              'Greed search%s'%opt.greed_search_range):
            threshold = threshold / 100.0

            nb_tp, nb_fp, nb_gt = 0, 0, 0
            # filtering bbox by threshold, only think one class here
            for k, value in tp_fp_dict.items():
                nb_tp += sum(np.array(value[0]['tp']).reshape((-1, 5))[:, 0] > threshold)
                nb_fp += sum(np.array(value[0]['fp']).reshape((-1, 5))[:, 0] > threshold)
                nb_gt += len(np.array(value[0]['gt']))

            nb_fn = nb_gt - nb_tp
            acc = nb_tp / (nb_gt + nb_fp)
            recall, precision = nb_tp / (nb_tp + nb_fn + 1e-16), nb_tp / (nb_tp + nb_fp + 1e-16)
            f1_score = 2 * precision * recall * 1.0 / (recall + precision + 1e-16)


            if f1_score > best_f1:
                best_acc = acc
                best_threshold = threshold
                best_f1 = f1_score

                Logger.info(f'threshold:{threshold}, tp:{nb_tp}, fp:{nb_fp}, fn:{nb_fn}, gt:{nb_gt}, acc:{round(acc, 2)}, '
                            f'recall:{round(recall, 2)}, precision:{round(precision, 2)}, best_f1:{round(best_f1, 2)}')



        for image_id, value in tp_fp_dict.items():
            # only think one class here, 0 is class_idx
            gts, fps, tps = value[0]['gt'], value[0]['fp'], value[0]['tp']
            json_templates[image_id].add_list_bboxes(gts, status = 'gt')
            json_templates[image_id].add_list_bboxes(fps, status = 'pred')
            json_templates[image_id].add_list_bboxes(tps, status = 'pred')

        if use_thread_pool:
            thread_pool = ThreadPool(10)
            Logger.debug('Using thread pool to write image and json file')
        else:
            thread_pool = None

        p_bar = tqdm(total = len(json_templates.keys()), desc="Export json with conf threshold:%s"%max(0.1,
                                                                                                       best_threshold))

        def write_img_json(item, best_threshold, p_bar):
            item.draw_pred_gt_bboxes(conf_threshold= best_threshold)
            item.write(conf_threhsold = max(0.1, best_threshold))

            # item.draw_pred_gt_bboxes(conf_threshold= 0.1)
            # item.write(conf_threhsold = max(0.1, 0.1))
            p_bar.update()

        for image_id, item in json_templates.items():

            kwds = {'item': item, 'best_threshold': best_threshold, 'p_bar': p_bar}
            if thread_pool is not None:
                thread_pool.apply_async(write_img_json, kwds = kwds)
            else:
                write_img_json(**kwds)
        if thread_pool is not None:
            thread_pool.close()
            thread_pool.join()

        with open((save_dir / "평가결과.txt").as_posix(), 'w', encoding='utf-8') as f:
            content = str({'F1-Score': best_f1,
                           'accuracy': best_acc, 'mAP':best_map})

            f.write(content)
            Logger.info('final map, f1, acc, map50:%.4f %.4f %.4f %.4f'%(best_map, best_f1, best_acc, map50))
    else:
        # correct, conf, pcls, tcls
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

    # Print results
    if training:
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        Logger.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')


    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='eval.py')

    parser.add_argument('--model_path', nargs='+', type=str,
                        default='/Strawberry/yolov5/output/train/exp_904aug_ndata/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='/home/jovyan/project/jbnu/pwd2022_params/pwd.yaml', help='*.data '
                                                                                                              'path')
    parser.add_argument('--input', type=str,
                        help='json input folder path')

    parser.add_argument('--voc', default='/home/jovyan/datasets',type=str)
    parser.add_argument('--workers', type=int, default=6, help='maximum number of dataloader workers')
    parser.add_argument('--output', default='output/test', help='save to project/name')



    # parser.add_argument('--model_path', nargs='+', type=str,
    #                     default='/home/khtt/code/insitute_demo/yolov5_deploy/project/jbnu/yolov5/weights/best_ap05.pt',
    #                     help='model.pt path(s)')
    #
    # parser.add_argument('--data', type=str, default='/home/khtt/code/insitute_demo/yolov5_deploy/project/jbnu/pwd2022_params/pwd.yaml', help='*.data '
    #                                                                                                           'path')
    # parser.add_argument('--input', type=str, default = '/dataset/khtt/dataset/pine2022/ECOM/2.labled/split_test_a',
    #                     help='json input folder path')
    #
    # parser.add_argument('--voc', default='/dataset/khtt/dataset/pine2022/ECOM/2.labled/3.generated/preprocess_split_test_a',type=str)
    #
    # parser.add_argument('--output', default='/dataset/khtt/dataset/pine2022/ECOM/7.evaluations/yolov5_split_test_a', help='save to '
    #                                                                                                    'project/name')
    # parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')


    parser.add_argument('--greed_search', action='store_false')
    parser.add_argument('--greed_search_range', nargs='+', type=int, default = [50, 51, 1])

    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--mode', type= str, default='eval', help='sign to denote current '
                                                                  'status-train/eval/inference/serve')
    parser.add_argument('--img_size', type=int, default=800, help='eval size (pixels)')
    parser.add_argument('--img_suffix', type=str, default='tif', help='image suffix')
    parser.add_argument('--threshold', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--IoU', type=float, default=0.5, help='overlap iou for evaluation')
    parser.add_argument('--task', default='test', help="'val', 'test', 'study'")
    parser.add_argument('--gpu_num', default='0, 1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_false', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--name', default='', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')


    opt = parser.parse_args()

    from datetime import datetime
    opt.output = '%s_%s'%(opt.output, datetime.now().strftime("%Y%m%d_%H%M%S"))


    opt.device, opt.weights, opt.iou_overlap, opt.conf_thres,  opt.project = opt.gpu_num, opt.model_path, \
                                                                             opt.IoU, opt.threshold, \
                                                                             opt.output


    opt.raw_input = opt.input
    opt = preprocess_raw_data(opt.input, opt, suffix=opt.img_suffix)
    voc2yolo_convert(opt.input, sets = ['val'])



    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file

    Logger.info('Version: {}'.format("1.1.1"), '='*50)
    Logger.info(opt)


    if opt.task in ['val', 'test']:  # run normally


        test(opt=opt, data=opt.data,
             weights = opt.weights,
             batch_size = opt.batch_size,
             imgsz = opt.img_size,
             conf_thres = opt.conf_thres,
             iou_overlap = opt.iou_overlap,
             save_json = opt.save_json,
             single_cls = opt.single_cls,
             augment = opt.augment,
             verbose = opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(f, x)  # plot
