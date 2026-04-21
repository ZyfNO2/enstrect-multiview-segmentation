# ENSTRECT 项目部署计划

## 环境概况

* **工作目录**: `G:\Zed\ENSTRECTtest`

* **Conda 路径**: `G:\Anaconda`

* **目标环境**: `zed` (Python 3.8, 已存在)

* **项目地址**: <https://github.com/ben-z-original/enstrect>

> ⚠️ **注意**: zed 环境为 Python 3.8，而 ENSTRECT 官方推荐 Python 3.10。将尝试在现有 zed 环境中部署，如遇严重兼容性问题再考虑升级。

***

## 部署步骤

### Phase 1: 环境准备与连通性测试

#### Step 1.1 - 激活 zed 环境并验证基础连通性

* 使用 `G:\Anaconda\envs\zed\python.exe` 直接运行或通过 conda activate

* 运行 `python --version` 确认 Python 版本

* 运行 `pip --version` 确认 pip 可用

* 运行简单 import 测试确认环境健康

#### Step 1.2 - 检查 zed 现有依赖

* 列出已安装的包: `pip list`

* 重点检查: torch, numpy, open3d (已看到 open3d 存在) 等关键依赖版本

### Phase 2: 项目克隆与安装

#### Step 2.1 - 克隆 ENSTRECT 仓库

```bash
git clone https://github.com/ben-z-original/enstrect.git
```

#### Step 2.2 - 安装 ENSTRECT (可编辑模式)

```bash
pip install -e .
```

* 处理可能的依赖冲突

* 记录安装过程中的警告和错误

#### Step 2.3 - PyTorch3D 安装 (难点⚠️)

* PyTorch3D 官方主要支持 Linux

* Windows 上可能需要从源码编译或使用非官方 wheel

* 备选方案:

  1. 尝试直接 `pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7`
  2. 若失败，检查是否有 Windows 兼容的预编译 wheel
  3. 最后手段：跳过 PyTorch3D，仅验证核心模块导入

### Phase 3: 连通性测试

#### Step 3.1 - 基础模块导入测试

```bash
python -c "import enstrect; print('ENSTRECT imported successfully')"
```

#### Step 3.2 - 分模块导入测试

* `enstrect.segmentation` 分割模块

* `enstrect.mapping` 映射模块

* `enstrect.extraction` 提取模块

#### Step 3.3 - 分割模型快速测试

```bash
python -m enstrect.segmentation.nnunet_s2ds
```

(会自动下载模型权重)

### Phase 4: 示例运行 (可选，需下载数据)

#### Step 4.1 - 下载示例数据

```bash
python -m enstrect.datasets.download
```

#### Step 4.2 - 运行完整 pipeline 示例

* 使用 Bridge B 或 Bridge G 测试数据

* 以低分辨率 (scale=0.25) 快速验证

***

## 风险点与应对策略

| 风险                         | 影响             | 应对策略          |
| -------------------------- | -------------- | ------------- |
| Python 3.8 vs 3.10         | 可能存在语法/API 不兼容 | 逐个错误修复或升级环境   |
| PyTorch3D Windows 兼容性      | 可能无法安装         | 寻找替代方案或部分功能降级 |
| CUDA/PyTorch 版本对齐          | GPU 加速可能不可用    | 先以 CPU 模式运行   |
| 网络问题 (GitHub/Google Drive) | 克隆/下载失败        | 使用代理或镜像源      |

***

## 预期输出

* [ ] zed 环境可以正常激活和运行 Python

* [ ] ENSTRECT 代码成功克隆到 `G:\Zed\ENSTRECTtest`

* [ ] `pip install -e .` 成功完成

* [ ] 核心模块可以成功 import

* [ ] 分割模型可以加载并运行

* [ ] (可选) 完整示例 pipeline 可以执行

