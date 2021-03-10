import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Tespit():


    def __init__(self):
       # self.name = name
       # self.koordinat = koordinat
        self.label_list = []
        self.koordinat_list = []

        #print("tespit edilen : " , self.name)
        #print("koordinatları : " , self.koordinat)




def detect(save_img=False):
    atama = 0
    flag1 = False
    flag2 = False
    flag3 = False


    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:        # her frame burda dönüyor
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                sonuc = Tespit()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):   # xyxy koordinatlar imiş
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)


                        # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        (plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3))
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()



                        ab = torch.tensor(xyxy).view(1,4)[0]
                        ab = ab.numpy()

                        isim = f'{names[int(cls)]}'
                        sonuc.label_list.append(isim)
                        sonuc.koordinat_list.append(ab)
                #sol cıkıs ve kalıp label bulma
                i = 0
                cıkıs_list = []
                while i < len(sonuc.label_list):
                    if('CIKIS' == sonuc.label_list[i]):
                        cıkıs_list.append(sonuc.koordinat_list[i])

                    if('KALIP' == sonuc.label_list[i]):
                        kalıp = sonuc.koordinat_list[i]

                    i += 1
                if(len(cıkıs_list) == 2):
                    if(cıkıs_list[0][0] < cıkıs_list[1][0]):
                        sol_cıkıs = cıkıs_list[0]
                    else:
                        sol_cıkıs = cıkıs_list[1]
                elif(len(cıkıs_list) == 1):
                    sol_cıkıs = cıkıs_list[0]
                else:
                    sol_cıkıs = [0,0,0,0]
                sol_alan = (sol_cıkıs[2] - sol_cıkıs[0]) * (sol_cıkıs[3] - sol_cıkıs[1])

                cv2.putText(im0, "SOL ALAN" + str(sol_alan), (800, 500), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (209, 80, 0, 255), 3)

                try:
                    x1 = kalıp[0]
                    x2 = kalıp[2]
                    y1 = kalıp[1]
                    y2 = kalıp[3]

                    print("x1 degeri : " , x1 , " x2 degeri : " , x2 , " y1 degeri : " , y1 , " y2 degeri : " , y2)
                    if ((640 < x1 < 675) and (855 < x2 < 874) and (290 < y1 < 305) and (835 < y2 < 855)) or flag1:
                        flag1 = True
                        print("butun kosullar saglandı")
                        cv2.putText(im0, "UYGUN KONUM" ,(200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (209,80, 0 ,255), 3)
                        #cıkıs = sonuc.koordinat_list[0]
                        #if (640 < cıkıs[0] < 700):
                            #alan = (cıkıs[2] - cıkıs[0]) * (cıkıs[3] - cıkıs[1])
                        #else:
                            #cıkıs = sonuc.koordinat_list[1]
                            #alan = (cıkıs[2] - cıkıs[0]) * (cıkıs[3] - cıkıs[1])
                        print("sol kapak alanı : " , str(sol_alan))
                        if(2000 < sol_alan < 2400) or flag2:
                            # print("cıkmıs")
                            flag2 = True
                            if (900 < sol_alan < 1200) or flag3:
                                # print("en aşşa indi")

                                flag3 = True
                                if (2000 < sol_alan):
                                    if 'BOS' in sonuc.label_list:
                                        cv2.putText(im0, "SIKINTI YOK", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (209, 80, 0, 255), 3)
                                        print("******************sıkıntı yok***********************")

                                    else:
                                        cv2.putText(im0, "KALIP DUSMEDI", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (209, 80, 0, 255), 3)
                                        print("bunu bi şekilde halletmemiz gerek")


                                    time.sleep(1)
                                    print("flag yazdırdık: "+ str(flag1)+ " "+ str(flag2)+ " "+ str(flag3))

                    else:
                        print("kosullar saglanmadı")


                    if ((sol_cıkıs[0] > 800) and flag3):
                        flag1 = False
                        flag2 = False
                        flag3 = False

                except IndexError:
                    pass
                # kalıp ın son durumdaki koordinatı : [        672,         302,         862,         840]


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            atama += (t2 - t1)
            print("meloo", str(atama))


            # Stream results
            if True:        # önceden view_img idi, şimdi True oldu yani resimleri video gibi oynatma sağlandı
                cv2.imshow("Result", im0)   # farklı isim olursa ayrı pencerelerde açılır, aynı isimle aynı pencerede açar
                cv2.waitKey(1)  # 1 millisecond  - 0 girilir ise oynaması için imagein input bekler -- 1 kalması yeterli bizim için

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
        #print("gecen zaman :" , str(t2-t1))
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')





    #print("kalıp koordinatı" , sonuc.koordinat_list[2])
    #print("kalıp x1 : " , sonuc.koordinat_list[2][0])
    #print("kalıp x2 : ", sonuc.koordinat_list[2][2])
    #print("kalıp y1 : ", sonuc.koordinat_list[2][1])
    #print("kalıp y2 : ", sonuc.koordinat_list[2][3])


    #cıkıs yönlerini bulma
    #i = 0
    # while i<9:
    #     print("cıkıs1 koordinatları" , sonuc.koordinat_list[i][0])
    #     print("cıkıs2 koordinatları", sonuc.koordinat_list[i+1][0])
    #     cıkıs1 = sonuc.koordinat_list[i][0]
    #     cıkıs2 = sonuc.koordinat_list[i+1][0]
    #     if(640 < cıkıs1 < 700):
    #         print("cıkıs1 soldadır")
    #     else:
    #         print("cıkıs1 sağdadır")
    #
    #     if(640 < cıkıs2 < 700):
    #         print("cıkıs2 soldadır")
    #     else:
    #         print("cıkıs2 sağdadır")
    #     i = i+3

    #çıkışların alanlarını bulma


    #sol çıkış için işlemler : sol çıkış ilk baştaki koordinatı -> [   667,   172,   700,    204]
    #sol çıkışın başlangıç alanı : 1056
    #sol çıkışın sondaki alanı : 2352
    #sağ çıkışın başlangıç alanı : 598
    #sağ çıkışın sondaki alanı : 1410




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()


    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                #atama += 1
                #print("meloo 1-- ", str(atama))
                strip_optimizer(opt.weights)
        else:
            detect()
            #atama += 1
           # print("meloo 2-- ", str(atama))
