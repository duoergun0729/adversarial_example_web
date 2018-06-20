## 对抗样本攻击实验平台 (Adversarial example attack platform）

### 平台功能

本平台使用 TensorFlow 官方的 Cleverhans 对抗攻击方法库;

基于 MNIST、CIFAR-10、ImageNet三种数据集;

使用多种对抗性攻击方法在线生成对抗样本，并可设置攻击参数和攻击目标。

### 平台特色

网页后端使用 Django 框架，前端使用 Vue.js 框架，使得前后端完全分离。

前端使用 Element UI，并使用了 [vue-element-admin](https://github.com/PanJiaChen/vue-element-admin) 模板

### 使用方法

下载模型 model 文件夹到工程根目录。

百度网盘链接:[https://pan.baidu.com/s/1cbhVZNPeaTFPCRmRlDuIXg](https://pan.baidu.com/s/1cbhVZNPeaTFPCRmRlDuIXg)  密码:kjir

Google Drive 链接: [https://drive.google.com/open?id=1zHa6AfcyN2dJfTJcNqlEfwxVo_1f4RrQ](https://drive.google.com/open?id=1zHa6AfcyN2dJfTJcNqlEfwxVo_1f4RrQ)

在 frontend 前端文件进行 Vue.js 初始化

#### Start

```
# install dependency
npm install

# develop
npm run dev
```

#### Build

```
# build for test environment
npm run build:sit

# build for production environment
npm run build:prod
```

Build 后可直接运行 Django 工程展示前端页面