# 数据集辅助脚本

本目录只保留生成标注的源码，不保存图片、XML、第三方框架或模型。脚本通过自身位置定位仓库根目录，默认使用以下外部目录：

```text
external/
├─ data/
│  ├─ images/{dragon,peak,soar}/
│  └─ annotations/{dragon,peak,soar}/
├─ models/
│  ├─ hagrid/YOLOv10n_hands.pt
│  └─ rtmpose/rtmpose-m_simcc-hand5.pth
└─ frameworks/mmpose/configs/hand_2d_keypoint/rtmpose/hand5/
```

## HaGRID bbox

`infer_hagrid_cvat_xml.py` 使用HaGRIDv2 YOLOv10n Hand Detector生成bbox，并写出CVAT 1.1 XML：

```bash
python dataset/infer_hagrid_cvat_xml.py --dataset peak
```

可通过 `--input-root`、`--output-root` 和 `--weights` 覆盖默认位置。Ultralytics应安装在当前Python环境中，不再加载服务器vendor目录。

## RTMPose关键点

`infer_rtmpose_cvat_xml.py` 读取对应标注目录中的 `annotations校验后.xml`，保持bbox中心不变，将ROI扩展为边长 `max(width,height) × 1.2` 的正方形并缩放到256×256，然后生成两个关键点及 `annotations_square_roi_rtmpose.xml`：

```bash
python dataset/infer_rtmpose_cvat_xml.py --dataset peak
```

可通过 `--image-root`、`--annotation-root`、`--config`、`--checkpoint` 和 `--output` 覆盖默认位置。MMPose及其依赖应安装在当前Python环境中。
