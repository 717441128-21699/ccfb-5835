# 病理全切片拼接系统(纯 Python + numpy/scipy)

用一套可解释、可调参的自研流水线替代商业拼接软件:从带重叠、位置仅大致已知的
显微镜视野出发,做特征匹配求相对位移、全局鲁棒优化消除累积误差、曝光补偿与
多频段融合消接缝,最后输出金字塔瓦片并用自带网页查看器平滑缩放平移。

**不依赖 OpenCV / SIFT / 任何图像拼接库**,特征、匹配、位姿图、融合、金字塔全部
基于 numpy/scipy 实现;仅用 Pillow 做图像编解码。

## 流水线

```
合成/真实大图
   │  cut_tiles:网格裁切 + 位置抖动 + 暗角 + 曝光差 + 噪声 + 随机缺失
   ▼
缺陷视野(PNG + manifest,真值位姿单独保存仅用于打分)
   │  features:Harris 角点 + 归一化 patch 描述子 + Lowe 比值/双向匹配
   │  matching:名义几何门控 → 中位数稳健共识 → NCC 亚像素细化
   │           特征不足时降采样粗搜 + 原图精修的局部 NCC 兜底
   ▼
相对平移边(4 邻域位姿图)
   │  pose:Huber 鲁棒 IRLS 稀疏最小二乘,一次求全部视野全局位姿
   │       (对照:BFS 链式累积,量化不做全局优化时的漂移)
   ▼
全局位姿
   │  exposure:重叠区增益最小二乘 + 中心亮度绝对锚定
   │  flatfield:逐瓦片低频暗角平坦化
   ▼
亮度一致的就位视野
   │  blending:hard / feather / Burt–Adelson 拉普拉斯金字塔多频段
   ▼
无缝拼接大图
   │  pyramid:多分辨率 256px JPEG 瓦片 + JSON manifest
   ▼
viewer/index.html 原生 JS 查看器(LOD 按需加载、惯性平移、光标缩放)
```

## 目录结构

```
stitcher/
  synthetic.py    合成 H&E 风格组织大图(可复现种子)
  tile_data.py    切缺陷视野(抖动/暗角/增益/噪声/缺失)
  features.py     Harris 角点 + 归一化描述子 + 配对
  matching.py     几何门控、稳健共识、NCC 局部搜索与亚像素细化
  pose.py         位姿图、Huber IRLS 全局优化、链式累积对照
  exposure.py     重叠区增益补偿 + 绝对曝光锚定
  flatfield.py    瓦片内暗角平坦化
  blending.py     hard / feather / 多频段拉普拉斯金字塔融合
  pyramid.py      多分辨率瓦片金字塔
  evaluation.py   位姿 RMSE、盲对齐图像误差、接缝不连续度
  io_utils.py     读写工具
scripts/
  run_demo.py     端到端编排
  serve.py        本地静态服务器(打开查看器)
tests/
  test_pipeline.py 关键算法自测
viewer/
  index.html      无依赖网页查看器
PARAMETERS.md     全部参数的调优指南与排错路径
```

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pillow

# 算法自测(秒级)
.venv/bin/python tests/test_pipeline.py

# 端到端:10×10=100 视野,4096 源图(约 40 秒)
.venv/bin/python scripts/run_demo.py --workdir out_demo --cols 10 --rows 10

# 打开查看器
.venv/bin/python scripts/serve.py out_demo/pyramid
# 浏览器访问 http://localhost:8000/
```

常用命令行参数(完整列表 `--help`,调参见 PARAMETERS.md):

```bash
--cols/--rows      视野网格规模
--tile             单视野像素(默认 512)
--overlap          名义重叠比例(默认 0.25)
--jitter           位置抖动 σ,像素(默认 18)
--missing          缺失视野比例(默认 0.04)
--bands            多频段金字塔层数(默认 5)
--skip-pyramid     只出拼接图,不切瓦片
```

## 输出产物

`out_demo/` 下:

- `source.png` 原始大图(评估基准)
- `tiles/` 带缺陷的模拟视野与 `manifest.json`(含真值,仅打分用)
- `mosaic_hard/feather/multiband.png` 三种融合结果
- `eval_*_diff*.png` 与原图的误差热力图
- `pyramid/` 瓦片金字塔 + `index.html` 查看器
- `report.json` 全部指标与配置

## 评估口径

- **位姿 RMSE**:去掉单一全局平移(拼接只恢复相对几何)后,估计原点与真值的
  欧氏误差;同时报告链式累积的 RMSE 作对照。
- **图像误差**:在不知道真值的前提下,用相位相关做盲对齐(整数粗对齐 + NCC
  亚像素细化),再在有效覆盖区内算 RMSE/MAE/PSNR,避免整数量化虚高指标。
- **接缝不连续度**:在每条边的重叠中线量两侧窄条带灰度跳变,并给出相对天然
  图像梯度的比值。

## 设计要点与取舍

- **几何门控优先于复杂描述子**:显微镜纹理重复度高,单纯靠描述子容易飞点。
  利用网格先验把候选位移限制在名义位置附近,再做共识,比堆特征更稳。
- **局部 NCC 兜底而非全局相位相关**:大抖动下全局相位相关会锁到错误的重复峰;
  在名义位置小窗口内两级搜索,既抗重复纹理又能覆盖 3σ 抖动。
- **Huber IRLS**:少数物理上几乎不可配的极端边不会被最小二乘放大成全图扭曲。
- **每频段独立归一化的拉普拉斯金字塔**:常量信号逐频段正确还原,过渡带宽与
  重叠尺度匹配,低频亮度差在粗层平滑、高频细节在细层保留。
- **盲对齐评估**:指标不依赖拼接流程内部信息,可直接用于真实(无真值)数据的
  自检闭环。
