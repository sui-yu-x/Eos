# Eos

该仓库是赛后代码总结，选择的赛题为第十届全国大学生集成电路创新创业大赛_AI芯片应用赛道_思特威赛题_场景二。

## 仓库任务说明

我负责 **“手掌”** / **“手部”** 检测模型的训练。训练最初，面向 **手掌** 检测任务，在总决赛阶段，将其更改为了 **手部** 检测任务。首先简述两者的区别：
**手掌** 检测任务：**关注形态相对稳定的掌部区域**，受手指姿态变化和局部遮挡影响较小，因此检测难度通常更低、边界框也更稳定。但其检测框无法直接覆盖完整手部，后续需要额外扩展 ROI 才能用于手部关键点检测。
**手部** 检测任务：对 **整个手部进行检测**，包含手掌和手指的完整区域。由于不同手势下手指伸展、弯曲和遮挡会导致手部整体形状、尺度和长宽比变化较大，整体变化更多样更复杂。此外，手指在图像中的像素占比小，更容易被模型忽略或受环境干扰。

所以实际上 **手部** 检测任务更复杂，需要更多样本用于训练。但本赛题采用 **SC132GS** 图像传感器作为采集设备，所以模型训练不采用公开数据集，而是自行拍摄并标注。开源的用于 **手掌** 检测任务的 **Google Mediapipe** 模型 **漏检误检** 现象严重，精度不够；而 **HaGRIDv2 YOLOv10n Hand Detector** 的漏检误检情况少很多，精度也更高。所以最终选择使用 **HaGRIDv2 YOLOv10n Hand Detector** 作为首轮粗标注(每一份图片都需要经过人工核验)，并且将 **手掌** 检测任务修改为 **手部** 检测任务。

还有一个值得注意的旋转操作与时间线：
1. 在初赛时，我首先使用开源数据集作为模型训练输入，开源数据集图片尺寸为 $1920 \times 1080$，而 **SC132GS** 采集的图片尺寸为 $720 \times 1280$，所以训练时我将传感器采集图片向右进行旋转，使得其与开源数据集图片的尺寸一致，这就推理时需要首先对传感器采集的图片进行 **旋转处理**，存在额外的计算开销。
2. 分赛区阶段，重新采用 **SC132GS** 采集的图片作为训练输入，训练时不再进行旋转处理，推理时也不需要进行旋转处理。但训练效果相当不好，可能原因是：为了使得在进行手语动作时，手部在图像中尽量完整地被采集到，拍摄时需要离镜头较远，导致 **手部在图像中占据的像素比例较小**，训练难度更高(在总决赛阶段有所体现，见[小手检测的难度](#小手检测的难度) )；并且拍摄时 **场景单一**，**手的变化少**，训练时 **逐帧采样** (相邻帧变化小)，最终模型训练时容易过拟合，泛化能力差。**最终仍旧采用初赛模型。**
3. 总决赛阶段，没有再次尝试使用$720 \times 1280$作为模型训练与推理输入，而是继续将其向右旋转，优化$1280 \times 720$尺寸下训练出的模型。致使 **开发板上需要大量时间用于图像旋转处理**，增加了推理时间开销。

所以，如果能够直接使用 $720 \times 1280$ 作为模型训练与推理输入，并且仍然采用 **手掌** 任务，可以极大地优化该模块。
另外，为了更好的学习手指的细粒度信息，我将模型输入尺寸由 $224 \times 224$ 增加至 $384 \times 224$，以减小整手检测的任务难度。但这导致预处理与推理延迟增加，最后一天，我临时训练一份 $224 \times 224$ 的模型，预处理推理延迟明显降低(37->17)，但在开发板上表现很差，未能采纳。该改变也能极大地优化该模快。
很遗憾我都未能做到。

---

## 小手检测的难度

总决赛阶段，数据集录制时，我们设置了7个变量：背景、距离、明暗、手势、训练验证测试、录制批次演示人员。背景包含complex和white；距离包含near、mid和far；明暗包含bright和dark；手势包含fist、flat、go、ok、one、open、random和thick；训练验证测试即train、val和test；录制批次如s01，s02等；演示人员包含dragon、peak和soar。训练过程中，明显发现模型在far情况下的推理结果更差：
在第五次训练时，以下两组数据集均包含在 **训练集** 中：white-far-bright-fist-train-s01-dragon与white-mid-bright-fist-train-s01-dragon。这两组数据集的差别仅为 **距离**，以这两组数据集为例，训练的最终结果如下所示：

| 文件夹 | 图片数量| P | R | F1 | bbox面积 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| white-far-bright-fist-train-s01-dragon | 600 | 0.2248 | 0.1600 | 0.1870 | 0.0115 |
| white-mid-bright-fist-train-s01-dragon | 600 | 0.9679 | 0.8042 | 0.8785 | 0.0263 |

**P是Precision，精准率**，指与真实框的IOU阈值大于0.5的预测框占总的预测框的比例；**R是Recall，召回率**，指与真实框的IOU阈值大于0.5的预测框占总的真实框的比例。
bbox面积指的是经过人工复核的**归一化面积中位数**，对于white-bright-fist-train-s01-dragon条件，mid条件下的归一化bbox面积约为far的2.29。mid条件的P和R明显优于far条件，其中P是far的5.03倍，R是far的4.70倍。

---

## 标注平台

在本次赛事中，使用了CVAT作为标注平台，全称为Computer Vision Annotation Tool。CVAT是一个开源的图像和视频标注工具，支持多种标注类型，包括边界框、分割、多边形、关键点等。它提供了一个用户友好的界面，使标注人员能够高效地进行标注工作，并支持团队协作和任务管理。网站地址为[CVAT](https://www.cvat.ai/)

---

## 训练代码与推理代码说明

本目录汇总第七次 Palm 检测器训练所使用的重要代码、本地推理源码与配置、数据集准备辅助代码，以及 HDF5、ONNX、量化数据和 M1 部署资产。HaGRID YOLOv10 与 RTMPose 未保存在本目录，README 提供从官方来源获取的方法。

### 1. 目录结构

```text
Eos/
├─ README.md
├─ dataset/
│  ├─ README.md
│  ├─ infer_hagrid_cvat_xml.py
│  └─ infer_rtmpose_cvat_xml.py
├─ train/
│  ├─ seventh_config.py
│  ├─ model.py
│  ├─ anchor_utils.py
│  ├─ anchor_config.json
│  ├─ losses.py
│  ├─ hard_sample_pool_seventh.py
│  ├─ train_seventh.py
│  ├─ run_seventh_train.sh
│  └─ best_model_six_train.weights.h5
├─ infer/
│  ├─ best_model.weights.h5
│  ├─ infer.py
│  ├─ model.py
│  ├─ anchor_utils.py
│  ├─ anchor_config.json
│  └─ recommended_inference.json
├─ onnx_files+dataset_for_quantization/
└─ m1model/
```

数据集、标注、训练划分和第三方模型不纳入仓库，运行时统一放在被 `.gitignore` 忽略的 `external/`：

```text
external/
├─ data/
│  ├─ images/{dragon,peak,soar}/
│  ├─ annotations/{dragon,peak,soar}/
│  └─ splits/seventh_train/
│     ├─ fixed_split_manifest.json
│     ├─ train.txt
│     ├─ legacy_val.txt
│     ├─ new_val.txt
│     └─ combined_val.txt
├─ models/
│  ├─ hagrid/YOLOv10n_hands.pt
│  └─ rtmpose/rtmpose-m_simcc-hand5.pth
└─ frameworks/mmpose/configs/hand_2d_keypoint/rtmpose/hand5/
```

### 2. 第七次训练代码说明

#### 2.1 入口脚本

- `run_seventh_train.sh`：正式训练入口。它从脚本位置推导仓库根目录，检查 `external/` 下的图片与固定划分；困难样本池不存在时会先自动生成，再以默认 batch size 96 启动训练。首次运行只加载第六次最佳权重作为模型初始化，再次运行只允许恢复第七次训练自己的 checkpoint。
- `best_model_six_train.weights.h5`：第六次最佳权重，位于 `train/`，是第七次训练的参数初始化来源。

#### 2.2 核心实现

- `seventh_config.py`：集中定义工作区、数据集、图像尺寸、初始化权重、随机种子、ATSS、困难样本和匹配阈值等训练约定。
- `model.py`：Palm 检测网络结构。模型为非 FPN 双尺度检测器，输出 `14×24` 与 `7×12` 两个尺度的回归和分类结果。
- `anchor_utils.py`：anchor 生成、ATSS 正样本分配、目标编码、预测解码和 NMS。
- `losses.py`：bbox 像素 Huber、GIoU、关键点像素 Huber 和 focal loss 的实现与组合。
- `train_seventh.py`：数据读取、横竖屏方向处理、增强、三数据集均衡采样、25% 困难样本采样、混合精度训练、验证指标、checkpoint、早停和训练状态保存。

#### 2.3 困难样本

`hard_sample_pool_seventh.py` 使用第六次权重扫描第七次训练集，在每个“数据集 × 文件夹”内选择约 20% 的困难图片并生成 `train/runs/seventh_train/hard_samples/hard_pool.json`。正式训练以 96 个样本为配额块，其中 72 个普通样本、24 个困难样本；可视化、人工复核和全量诊断输出不属于当前代码归档。

#### 2.4 服务器配置

在autodl上租的RTX 4090 * 1卡
镜像：PyTorch  1.11.0；Python  3.8(ubuntu20.04)；CUDA  11.3
GPU：RTX 4090(24GB) * 1
CPU：16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz

#### 2.5 服务器训练虚拟环境

第七次训练原环境为 Python 3.10.20 与 TensorFlow 2.10.0。启动脚本默认使用当前环境的 `python3`，也可以通过 `PYTHON` 环境变量指定解释器。原环境核验到的重要包如下：

| 包 | 版本 | 用途 |
| --- | --- | --- |
| Python | 3.10.20 | 训练与评估脚本运行环境 |
| TensorFlow | 2.10.0 | 模型构建、GPU 训练、自动求导、checkpoint 与推理 |
| Keras | 2.10.0 | 网络层、模型和 HDF5 权重接口 |
| NumPy | 1.24.4 | anchor、bbox、关键点和指标的数组计算 |
| opencv-python | 4.8.0.74 | TIFF/JPEG 解码、灰度预处理、增强和可视化 |
| tqdm | 4.67.3 | 数据审计、训练外评估等流程的进度显示 |
| h5py | 3.16.0 | Keras HDF5 权重文件读写支持 |
| TensorBoard | 2.10.1 | TensorFlow 配套的日志与可视化组件 |
| tensorflow-estimator | 2.10.0 | TensorFlow 2.10 配套组件 |
| protobuf | 3.19.6 | TensorFlow/TensorBoard 的序列化依赖 |

训练脚本还使用 Python 标准库以及系统命令 `bash`、`flock` 和 `nvidia-smi`。CUDA、cuDNN 和 NVIDIA 驱动属于服务器 GPU 运行栈，不属于该 Python 虚拟环境中的 pip 包。

### 3. 推理代码说明

- `infer.py`：本地批量推理入口。递归扫描输入目录，完成图像解码、方向校正、灰度预处理、模型推理、预测解码、NMS、结果绘制，并输出 JPEG 和 `inference_results.json`。
- `model.py`：与训练端一致的网络结构定义。
- `anchor_utils.py`：与模型输出配套的 anchor 和解码逻辑。
- `anchor_config.json`：两级特征图(14 $\times$ 14和7 $\times$ 7)所使用的 anchor 配置。
- `recommended_inference.json`：推荐的输入规格与阈值。
- `best_model.weights.h5`：第七次训练的最佳模型权重，也是 `infer.py` 的默认权重文件。

当前推理约定如下：

- 模型输入：`NCHW = 1×1×224×384`，灰度，归一化到 `[0,1]`。
- 原图为 `720×1280` 时顺时针旋转 90°；原图为 `1280×720` 时不旋转；其他尺寸会直接报错。
- 置信度阈值：`0.20`。
- NMS IoU 阈值：`0.10`。
- 每张图最多保留 `2` 个检测结果。
- anchor 总数：`840`。
- 模型参数量检查值：`1,367,620`。
- 可视化中 bbox 和置信度文字为绿色，两个辅助关键点为红色。

`infer.py` 支持 TensorFlow 2.10/Keras 2 的原生 HDF5 权重加载，也包含 Keras 3 旧版 HDF5 兼容加载路径。基本依赖包括 TensorFlow、NumPy、OpenCV；使用 Keras 3 兼容加载时还需要 `h5py`。

### 4. ONNX 与 M1 部署模型

#### 4.1 ONNX 模型

`onnx_files+dataset_for_quantization/model_1x1x384x224_opt.onnx` 是 Palm 检测器的优化 ONNX 模型。虽然文件名写作 `384x224`，图中的实际张量维度按 NCHW 表示：

- 输入：`inputs = [1, 1, 224, 384]`。
- `conv2d_46 = [1, 16, 14, 24]`：高分辨率层回归输出。
- `activation_1 = [1, 2, 14, 24]`：高分辨率层分类输出。
- `conv2d_41 = [1, 16, 7, 12]`：低分辨率层回归输出。
- `activation = [1, 2, 7, 12]`：低分辨率层分类输出。

该文件使用 ONNX opset 11，当前备份已通过 ONNX 模型结构检查。ONNX 文件只包含网络前向图；图像方向校正、灰度缩放、anchor 解码、置信度筛选和 NMS 仍需复用 `infer/` 中的前后处理约定。

#### 4.2 M1 模型及评估数据

`m1model/` 保存由芯片工具链生成的量化部署模型及仿真对比结果：

- `*.m1model`：最终 M1 部署模型。
- `*_InputOrderScale.txt`：输入张量顺序和量化 scale；当前输入 `inputs` 的 scale 为 `0.0039215689`。
- `*_OutputOrderScale.txt`：四个输出张量在 M1 模型中的顺序和各自量化 scale。
- `*_evaluate_report.json`：10 个样本的原始模型与 M1 仿真输出余弦相似度汇总。
- `img_*.d/*.ori.npy` 与 `*.sim.npy`：每个评估样本、每个输出头的原始结果和 M1 仿真结果，便于逐张定位量化误差。

评估报告中的四个输出平均余弦相似度分别为：`conv2d_46=0.992082`、`activation_1=0.957502`、`conv2d_41=0.995495`、`activation=0.973663`。这些数值只衡量输出张量的一致性，不等同于检测 Precision、Recall、F1 或 IoU。

### 5. 从官方地址获取 HaGRID YOLOv10（本仓库未包含）

`dataset/infer_hagrid_cvat_xml.py` 使用 HaGRIDv2 官方发布的 YOLOv10n 手部检测模型生成 bbox。本目录不保存 HaGRID YOLOv10 权重或附加运行库，需要时应从官方来源获取。

官方资料：

- HaGRIDv2 官方仓库：<https://github.com/hukenovs/hagrid>
- Ultralytics 官方安装文档：<https://docs.ultralytics.com/quickstart/>
- Ultralytics YOLOv10 文档：<https://docs.ultralytics.com/models/yolov10/>
- PyTorch 安装选择器：<https://pytorch.org/get-started/locally/>

#### 5.1 安装 Ultralytics

先按照 PyTorch 官方安装选择器安装与操作系统及 CUDA/CPU 环境匹配的 PyTorch，再安装 Ultralytics。为保持与本项目脚本适配时使用的 YOLOv10 API 一致，建议使用 8.2.91：

```bash
python -m pip install "ultralytics==8.2.91"
```

如果不要求复现固定版本，也可以按照 Ultralytics 官方文档安装当前稳定版：

```bash
python -m pip install -U ultralytics
```

#### 5.2 下载 HaGRIDv2 官方权重

本项目 bbox 脚本需要 `YOLOv10n_hands.pt`。HaGRID 官方仓库的模型表将该文件作为 YOLOv10n Hand Detector 发布。

Linux、macOS 或 Git Bash：

```bash
mkdir -p external/models/hagrid
curl -L \
  -o external/models/hagrid/YOLOv10n_hands.pt \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Path .\external\models\hagrid -Force
Invoke-WebRequest `
  -Uri "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt" `
  -OutFile ".\external\models\hagrid\YOLOv10n_hands.pt"
```

如需手势类别检测而非单纯手部 bbox，可从 HaGRID 官方模型表下载可选的 `YOLOv10n_gestures.pt`：

```text
https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_gestures.pt
```

`YOLOv10n_hands.pt` 与 `YOLOv10n_gestures.pt` 任务不同。当前 `infer_hagrid_cvat_xml.py` 只需要前者，不应将手势模型误作手部 bbox 默认权重。

#### 5.3 使用官方权重运行本项目脚本

通过 `--weights` 显式指定下载后的手部检测权重：

```bash
python dataset/infer_hagrid_cvat_xml.py --dataset dragon
```

默认输入为 `external/data/images/dragon`，输出为 `external/data/annotations/dragon`；也可通过 `--input-root`、`--output-root` 和 `--weights` 覆盖。

### 6. 从官方地址获取 RTMPose（本仓库未包含）

`dataset/infer_rtmpose_cvat_xml.py` 使用 OpenMMLab MMPose 的 RTMPose-M Hand5 模型生成两个辅助关键点。本目录不保存 RTMPose 框架、配置或 checkpoint，需要时应从官方来源获取。

官方资料：

- MMPose 仓库：<https://github.com/open-mmlab/mmpose>
- MMPose 安装文档：<https://mmpose.readthedocs.io/en/latest/installation.html>
- MMPose 推理文档：<https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md>
- PyTorch 安装选择器：<https://pytorch.org/get-started/locally/>

#### 6.1 安装 MMPose

先按照 PyTorch 官方安装选择器安装与操作系统、CUDA 或 CPU 环境匹配的 PyTorch 和 TorchVision。然后安装 OpenMMLab 基础组件：

```bash
python -m pip install -U openmim
mim install "mmengine>=0.9.0"
mim install "mmcv>=2.0.1"
```

再从 MMPose 官方仓库获取 1.3.2 版本并安装：

```bash
git clone --branch v1.3.2 --depth 1 https://github.com/open-mmlab/mmpose.git external/frameworks/mmpose
python -m pip install -e external/frameworks/mmpose
```

MMPose 1.x 应配合 MMCV 2.x 使用。当前脚本直接使用人工校验 bbox 作为 ROI，不需要额外目标检测器；只有运行依赖检测器的官方 demo 时才需要另外安装 MMDetection。

#### 6.2 下载官方 Hand5 checkpoint

Linux、macOS 或 Git Bash：

```bash
mkdir -p external/models/rtmpose
curl -L \
  -o external/models/rtmpose/rtmpose-m_simcc-hand5.pth \
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Path .\external\models\rtmpose -Force
Invoke-WebRequest `
  -Uri "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth" `
  -OutFile ".\external\models\rtmpose\rtmpose-m_simcc-hand5.pth"
```

对应配置文件包含在克隆后的 MMPose 官方仓库中：

```text
external/frameworks/mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py
```

不要只下载这一份 `.py` 配置，因为它还会引用 MMPose 仓库中的公共基础配置；应保留完整仓库或完整安装包。

#### 6.3 使用官方文件运行本项目脚本

通过 `--config` 和 `--checkpoint` 显式指定官方配置与权重位置：

```bash
python dataset/infer_rtmpose_cvat_xml.py --dataset dragon
```

### 7. 推理运行示例

`infer/` 已包含第七次最佳权重，因此保持当前目录结构时可以直接使用默认权重：

```powershell
python .\infer\infer.py `
  --input-dir .\external\data\images\dragon `
  --output-dir .\infer\images_palm `
  --config .\infer\recommended_inference.json
```

也可以通过 `--weights` 指定其他兼容权重，通过 `--confidence`、`--nms` 和 `--max-detections` 临时覆盖配置文件中的推理阈值。

### 8. 复现注意事项

1. 所有默认地址从脚本位置推导，不依赖当前工作目录。仓库不保存 `external/`；运行前必须按目录约定准备图片、标注、固定划分和第三方模型。
2. 划分文本和 manifest 中的图片地址必须相对 `external/data/images/`，格式为 `dragon/<source-folder>/<image>`、`peak/...` 或 `soar/...`，不接受绝对地址，例如/root/autodl-tmp/images/8_1/dragon/scene01/img_0001.jpg。
3. `run_seventh_train.sh` 使用 `BATCH_SIZE=96` 和当前环境的 `python3`；可分别通过 `BATCH_SIZE`、`WORKERS`、`HARD_BATCH_SIZE`、`PYTHON` 环境变量覆盖。困难样本池不存在时会自动生成。
4. 正式训练的模型初始化来自第六次最佳权重，但 optimizer、epoch 和训练状态重新开始；`--resume` 只用于恢复第七次训练自身的 checkpoint。
5. 推理代码会检查必需文件、推理配置、模型结构、参数量以及权重是否能被当前模型正确加载。
6. ONNX 和 M1 文件是部署资产，不能代替 `infer/` 中的方向处理、归一化、anchor 解码和 NMS。
7. HaGRID、RTMPose及其框架不纳入仓库，应按第5、6节放入约定的 `external/` 子目录。
