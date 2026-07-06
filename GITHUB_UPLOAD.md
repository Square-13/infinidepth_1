# GitHub Upload Guide

下面是在服务器或本机把本目录上传到 GitHub 的常用命令。

## 第一次上传

```bash
cd InfiniDepth_GitHub_Ready
git init
git add .
git commit -m "Add final far-depth inference version"
git branch -M main
git remote add origin git@github.com:YOUR_NAME/YOUR_REPO.git
git push -u origin main
```

如果使用 HTTPS：

```bash
git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
```

## 上传前检查

```bash
git status
git ls-files | grep checkpoints
find . -type f -size +50M
```

正常情况下不应该提交 `checkpoints`，也不应该出现超过 50MB 的大文件。

## 英文术语

`git init`：初始化一个 Git 仓库。

`git add`：把文件加入提交暂存区。

`git commit`：生成一次代码提交。

`git remote`：配置远程仓库地址。

`git push`：把本地提交推送到 GitHub。

