# Readme

## How to run
1. Clone to local repo
2. python main.py

`packages required`:
python3, cv2

## Seam folder structure
```
seam/
    |--in/：Input Images
    |   |--images/: Images to transform
    |	|--masks/: Masks for object removal
    |--out/: Output Images
    |   |--images/: The output images
    |   |--output.gif/: Combine output images to gif
    |main.py/: Run Seam Carving
    |seam_carving.py/: Implementation of Seam Carving algorithm
    |make_gif.py/: Combine images into gif
    |README.md/: shit
```

## Seam Carving

### 复现结果\
**Original Image**\
![original image](https://github.com/WeixiongLin/newshit/blob/main/figures/pic.jpg)


**Mask**\
![original image](https://github.com/WeixiongLin/newshit/blob/main/figures/mask.jpg)

**Output Image**\
![original image](https://github.com/WeixiongLin/newshit/blob/main/figures/out.gif)


*Notes:* 如果图片稍微大一点的话(3400x4000), seam carving 要处理非常长的时间, 那么 bottleneck 在哪里?

问题出在: seam_carving.cumulative_map_backward() 这个函数上.
该函数的作用是进行动态规划.

图片大就慢是因为 DP 方程需要计算的内容变多, OPT[m, n] += min{OPT[m-1, n-1], OPT[m-1, n], OPT[m-1, n+1]}
我想到对 DP 的改进就是每次选多条 seam delete, 否则每次计算了OPT然后再重新算太浪费了.

### 算法原理

1. Seam Carving 算法的作用是在尽量不改变图像内容的情况下对图片进行 resize

例如:\
**Original Image**\
![original image](https://github.com/vivianhylee/seam-carving/raw/master/example/image6.jpg)

**Resized Image**\
![image size expansion](https://github.com/vivianhylee/seam-carving/raw/master/example/image17_result.png)

2. Seam Carving 对一张图片计算 Energy, 纹理越复杂的地方 像素的 energy 越大
3. 如果希望图片的高度变矮, 则沿着水平方向选择 energy 低的像素进行删除

4. Object Removal: 实现 Object Removal 的效果需要两步
    1. 根据提供的图片 mask, 将 mask 对应区域的像素值设为 low, 然后进行 seam removal.
    2. 然后利用 seam carving 的 resize 功能, 将图片 resize 到原来水平.


1. [知乎专栏: Seam Carving](https://zhuanlan.zhihu.com/p/38974520?utm_source=tuicool&utm_medium=referral)
2. [算法论文](http://graphics.cs.cmu.edu/courses/15-463/2007_fall/hw/proj2/imret.pdf)
3. [github 复现参考](https://github.com/vivianhylee/seam-carving)

## Make gif
```
python make_gif.py
```
把指定文件夹下的图片合成为 gif.

*Note*: 文件夹下的图片名称按照时间顺序从小到大排列


## ROI

传统 Seam Carving 的问题在于:
1. 需要用DP计算 seam
2. mask 的设置必须非常精确, 否则会有残余

传统 ROI 方法的问题在于:
总的来说分辨率没有保证, 而且会出现马赛克

<!-- 我们可以先进行 image segmentation, 然后让用户轻易选择需要 remove 的 object. -->
