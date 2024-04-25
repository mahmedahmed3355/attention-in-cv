




##[***Attention***](#attention-series),[***Backbone***](#backbone-series), [***MLP***](#mlp-series), [***Re-parameter***](#re-parameter-series), [**Convolution**](#convolution-series)


![](https://img.shields.io/badge/python->=v3.0-blue)
![](https://img.shields.io/badge/pytorch->=v1.4-red)




**pytorch Attention CV**




-------

##

-  pip 

  pip 

  ```shell
  pip install Attention-pytorch
  ```




  ```shell
  git clone https://github.com/moh2236945/attention-in-cv.git


  ```

### 

#### pip 
```python
import torch
from torch import nn
from torch.nn import functional as F

#  pip 

from Attention-pytorch.attention.MobileViTv2Attention import *

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
```



- [Attention Series](#attention-series)
    - [1. External Attention Usage](#1-external-attention-usage)

    - [2. Self Attention Usage](#2-self-attention-usage)

    - [3. Simplified Self Attention Usage](#3-simplified-self-attention-usage)

    - [4. Squeeze-and-Excitation Attention Usage](#4-squeeze-and-excitation-attention-usage)

    - [5. SK Attention Usage](#5-sk-attention-usage)

    - [6. CBAM Attention Usage](#6-cbam-attention-usage)

    - [7. BAM Attention Usage](#7-bam-attention-usage)
    
    - [8. ECA Attention Usage](#8-eca-attention-usage)

    - [9. DANet Attention Usage](#9-danet-attention-usage)

    - [10. Pyramid Split Attention (PSA) Usage](#10-Pyramid-Split-Attention-Usage)

    - [11. Efficient Multi-Head Self-Attention(EMSA) Usage](#11-Efficient-Multi-Head-Self-Attention-Usage)

    - [12. Shuffle Attention Usage](#12-Shuffle-Attention-Usage)
    
    - [13. MUSE Attention Usage](#13-MUSE-Attention-Usage)
  
    - [14. SGE Attention Usage](#14-SGE-Attention-Usage)

    - [15. A2 Attention Usage](#15-A2-Attention-Usage)

    - [16. AFT Attention Usage](#16-AFT-Attention-Usage)

    - [17. Outlook Attention Usage](#17-Outlook-Attention-Usage)

    - [18. ViP Attention Usage](#18-ViP-Attention-Usage)

    - [19. CoAtNet Attention Usage](#19-CoAtNet-Attention-Usage)

    - [20. HaloNet Attention Usage](#20-HaloNet-Attention-Usage)

    - [21. Polarized Self-Attention Usage](#21-Polarized-Self-Attention-Usage)

    - [22. CoTAttention Usage](#22-CoTAttention-Usage)

    - [23. Residual Attention Usage](#23-Residual-Attention-Usage)
  
    - [24. S2 Attention Usage](#24-S2-Attention-Usage)

    - [25. GFNet Attention Usage](#25-GFNet-Attention-Usage)

    - [26. Triplet Attention Usage](#26-TripletAttention-Usage)

    - [27. Coordinate Attention Usage](#27-Coordinate-Attention-Usage)

    - [28. MobileViT Attention Usage](#28-MobileViT-Attention-Usage)

    - [29. ParNet Attention Usage](#29-ParNet-Attention-Usage)

    - [30. UFO Attention Usage](#30-UFO-Attention-Usage)

    - [31. ACmix Attention Usage](#31-Acmix-Attention-Usage)
  
    - [32. MobileViTv2 Attention Usage](#32-MobileViTv2-Attention-Usage)

    - [33. DAT Attention Usage](#33-DAT-Attention-Usage)

    - [34. CrossFormer Attention Usage](#34-CrossFormer-Attention-Usage)

    - [35. MOATransformer Attention Usage](#35-MOATransformer-Attention-Usage)

    - [36. CrissCrossAttention Attention Usage](#36-CrissCrossAttention-Attention-Usage)

    - [37. Axial_attention Attention Usage](#37-Axial_attention-Attention-Usage)

    - [38. HAM: Hybrid attention module in deep convolutional neural Network(#38-HAM)

    - [39. Global_Attention_Mechansim (#39.GAM)

- [Backbone Series](#Backbone-series)

    - [1. ResNet Usage](#1-ResNet-Usage)

    - [2. ResNeXt Usage](#2-ResNeXt-Usage)

    - [3. MobileViT Usage](#3-MobileViT-Usage)

    - [4. ConvMixer Usage](#4-ConvMixer-Usage)

    - [5. ShuffleTransformer Usage](#5-ShuffleTransformer-Usage)

    - [6. ConTNet Usage](#6-ConTNet-Usage)

    - [7. HATNet Usage](#7-HATNet-Usage)

    - [8. CoaT Usage](#8-CoaT-Usage)

    - [9. PVT Usage](#9-PVT-Usage)

    - [10. CPVT Usage](#10-CPVT-Usage)

    - [11. PIT Usage](#11-PIT-Usage)

    - [12. CrossViT Usage](#12-CrossViT-Usage)

    - [13. TnT Usage](#13-TnT-Usage)

    - [14. DViT Usage](#14-DViT-Usage)

    - [15. CeiT Usage](#15-CeiT-Usage)

    - [16. ConViT Usage](#16-ConViT-Usage)

    - [17. CaiT Usage](#17-CaiT-Usage)

    - [18. PatchConvnet Usage](#18-PatchConvnet-Usage)

    - [19. DeiT Usage](#19-DeiT-Usage)

    - [20. LeViT Usage](#20-LeViT-Usage)

    - [21. VOLO Usage](#21-VOLO-Usage)
    
    - [22. Container Usage](#22-Container-Usage)

    - [23. CMT Usage](#23-CMT-Usage)

    - [24. EfficientFormer Usage](#24-EfficientFormer-Usage)

    - [25. ConvNeXtV2 Usage](#25-ConvNeXtV2-Usage)



- [MLP Series](#mlp-series)

    - [1. RepMLP Usage](#1-RepMLP-Usage)

    - [2. MLP-Mixer Usage](#2-MLP-Mixer-Usage)

    - [3. ResMLP Usage](#3-ResMLP-Usage)

    - [4. gMLP Usage](#4-gMLP-Usage)

    - [5. sMLP Usage](#5-sMLP-Usage)

    - [6. vip-mlp Usage](#6-vip-mlp-Usage)

- [Re-Parameter(ReP) Series](#Re-Parameter-series)

    - [1. RepVGG Usage](#1-RepVGG-Usage)

    - [2. ACNet Usage](#2-ACNet-Usage)

    - [3. Diverse Branch Block(DDB) Usage](#3-Diverse-Branch-Block-Usage)

- [Convolution Series](#Convolution-series)

    - [1. Depthwise Separable Convolution Usage](#1-Depthwise-Separable-Convolution-Usage)

    - [2. MBConv Usage](#2-MBConv-Usage)

    - [3. Involution Usage](#3-Involution-Usage)

    - [4. DynamicConv Usage](#4-DynamicConv-Usage)

    - [5. CondConv Usage](#5-CondConv-Usage)

***



# Attention Series

- Pytorch implementation of ["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks---arXiv 2021.05.05"](https://arxiv.org/abs/2105.02358)

- Pytorch implementation of ["Attention Is All You Need---NIPS2017"](https://arxiv.org/pdf/1706.03762.pdf)

- Pytorch implementation of ["Squeeze-and-Excitation Networks---CVPR2018"](https://arxiv.org/abs/1709.01507)

- Pytorch implementation of ["Selective Kernel Networks---CVPR2019"](https://arxiv.org/pdf/1903.06586.pdf)

- Pytorch implementation of ["CBAM: Convolutional Block Attention Module---ECCV2018"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

- Pytorch implementation of ["BAM: Bottleneck Attention Module---BMCV2018"](https://arxiv.org/pdf/1807.06514.pdf)

- Pytorch implementation of ["ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks---CVPR2020"](https://arxiv.org/pdf/1910.03151.pdf)

- Pytorch implementation of ["Dual Attention Network for Scene Segmentation---CVPR2019"](https://arxiv.org/pdf/1809.02983.pdf)

- Pytorch implementation of ["EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network---arXiv 2021.05.30"](https://arxiv.org/pdf/2105.14447.pdf)

- Pytorch implementation of ["ResT: An Efficient Transformer for Visual Recognition---arXiv 2021.05.28"](https://arxiv.org/abs/2105.13677)

- Pytorch implementation of ["SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS---ICASSP 2021"](https://arxiv.org/pdf/2102.00240.pdf)

- Pytorch implementation of ["MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning---arXiv 2019.11.17"](https://arxiv.org/abs/1911.09483)

- Pytorch implementation of ["Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks---arXiv 2019.05.23"](https://arxiv.org/pdf/1905.09646.pdf)

- Pytorch implementation of ["A2-Nets: Double Attention Networks---NIPS2018"](https://arxiv.org/pdf/1810.11579.pdf)


- Pytorch implementation of ["An Attention Free Transformer---ICLR2021 (Apple New Work)"](https://arxiv.org/pdf/2105.14103v1.pdf)


- Pytorch implementation of [VOLO: Vision Outlooker for Visual Recognition---arXiv 2021.06.24"](https://arxiv.org/abs/2106.13112) 
  (https://zhuanlan.zhihu.com/p/385561050)


- Pytorch implementation of [Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition---arXiv 2021.06.23](https://arxiv.org/abs/2106.12368) 
  (https://mp.weixin.qq.com/s/5gonUQgBho_m2O54jyXF_Q)


- Pytorch implementation of [CoAtNet: Marrying Convolution and Attention for All Data Sizes---arXiv 2021.06.09](https://arxiv.org/abs/2106.04803) 
  (https://zhuanlan.zhihu.com/p/385578588)


- Pytorch implementation of [Scaling Local Self-Attention for Parameter Efficient Visual Backbones---CVPR2021 Oral](https://arxiv.org/pdf/2103.12731.pdf)  [【论文解析】](https://zhuanlan.zhihu.com/p/388598744)



- Pytorch implementation of [Polarized Self-Attention: Towards High-quality Pixel-wise Regression---arXiv 2021.07.02](https://arxiv.org/abs/2107.00782)  [【论文解析】](https://zhuanlan.zhihu.com/p/389770482) 


- Pytorch implementation of [Contextual Transformer Networks for Visual Recognition---arXiv 2021.07.26](https://arxiv.org/abs/2107.12292)  [【论文解析】](https://zhuanlan.zhihu.com/p/394795481) 


- Pytorch implementation of [Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021](https://arxiv.org/abs/2108.02456) 


- Pytorch implementation of [S²-MLPv2: Improved Spatial-Shift MLP Architecture for Vision---arXiv 2021.08.02](https://arxiv.org/abs/2108.01072) [【论文解析】](https://zhuanlan.zhihu.com/p/397003638) 

- Pytorch implementation of [Global Filter Networks for Image Classification---arXiv 2021.07.01](https://arxiv.org/abs/2107.00645) 

- Pytorch implementation of [Rotate to Attend: Convolutional Triplet Attention Module---WACV 2021](https://arxiv.org/abs/2010.03045) 

- Pytorch implementation of [Coordinate Attention for Efficient Mobile Network Design ---CVPR 2021](https://arxiv.org/abs/2103.02907)

- Pytorch implementation of [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2021.10.05](https://arxiv.org/abs/2110.02178)

- Pytorch implementation of [Non-deep Networks---ArXiv 2021.10.20](https://arxiv.org/abs/2110.07641)

- Pytorch implementation of [UFO-ViT: High Performance Linear Vision Transformer without Softmax---ArXiv 2021.09.29](https://arxiv.org/abs/2109.14382)

- Pytorch implementation of [Separable Self-attention for Mobile Vision Transformers---ArXiv 2022.06.06](https://arxiv.org/abs/2206.02680)

- Pytorch implementation of [On the Integration of Self-Attention and Convolution---ArXiv 2022.03.14](https://arxiv.org/pdf/2111.14556.pdf)

- Pytorch implementation of [CROSSFORMER: A VERSATILE VISION TRANSFORMER HINGING ON CROSS-SCALE ATTENTION---ICLR 2022](https://arxiv.org/pdf/2108.00154.pdf)

- Pytorch implementation of [Aggregating Global Features into Local Vision Transformer](https://arxiv.org/abs/2201.12903)

- Pytorch implementation of [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721)

- Pytorch implementation of [Axial Attention in Multidimensional Transformers](https://arxiv.org/abs/1912.12180)
***


1. External Attention Usage
1.1. Paper
["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"](https://arxiv.org/abs/2105.02358)


***


### 2. Self Attention Usage
#### 2.1. Paper
["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)

#### 1.2. Overview
![](./model/img/SA.png)

#### 1.3. Usage Code
```python
from model.attention.SelfAttention import ScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)
```

***

### 3. Simplified Self Attention Usage
#### 3.1. Paper
[None]()

#### 3.2. Overview
![](./model/img/SSA.png)

#### 3.3. Usage Code
```python
from model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
output=ssa(input,input,input)
print(output.shape)

```

***

### 4. Squeeze-and-Excitation Attention Usage
#### 4.1. Paper
["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507)

#### 4.2. Overview
![](./model/img/SE.png)

#### 4.3. Usage Code
```python
from model.attention.SEAttention import SEAttention
import torch

input=torch.randn(50,512,7,7)
se = SEAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)

```

***

### 5. SK Attention Usage
#### 5.1. Paper
["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf)

#### 5.2. Overview
![](./model/img/SK.png)

#### 5.3. Usage Code
```python
from model.attention.SKAttention import SKAttention
import torch

input=torch.randn(50,512,7,7)
se = SKAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)

```
***

### 6. CBAM Attention Usage
#### 6.1. Paper
["CBAM: Convolutional Block Attention Module"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

#### 6.2. Overview
![](./model/img/CBAM1.png)

![](./model/img/CBAM2.png)

#### 6.3. Usage Code
```python
from model.attention.CBAM import CBAMBlock
import torch

input=torch.randn(50,512,7,7)
kernel_size=input.shape[2]
cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
output=cbam(input)
print(output.shape)

```

***

### 7. BAM Attention Usage
#### 7.1. Paper
["BAM: Bottleneck Attention Module"](https://arxiv.org/pdf/1807.06514.pdf)

#### 7.2. Overview
![](./model/img/BAM.png)

#### 7.3. Usage Code
```python
from model.attention.BAM import BAMBlock
import torch

input=torch.randn(50,512,7,7)
bam = BAMBlock(channel=512,reduction=16,dia_val=2)
output=bam(input)
print(output.shape)

```

***

### 8. ECA Attention Usage
#### 8.1. Paper
["ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"](https://arxiv.org/pdf/1910.03151.pdf)

#### 8.2. Overview
![](./model/img/ECA.png)

#### 8.3. Usage Code
```python
from model.attention.ECAAttention import ECAAttention
import torch

input=torch.randn(50,512,7,7)
eca = ECAAttention(kernel_size=3)
output=eca(input)
print(output.shape)

```

***

### 9. DANet Attention Usage
#### 9.1. Paper
["Dual Attention Network for Scene Segmentation"](https://arxiv.org/pdf/1809.02983.pdf)

#### 9.2. Overview
![](./model/img/danet.png)

#### 9.3. Usage Code
```python
from model.attention.DANet import DAModule
import torch

input=torch.randn(50,512,7,7)
danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
print(danet(input).shape)

```

***

### 10. Pyramid Split Attention Usage

#### 10.1. Paper
["EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network"](https://arxiv.org/pdf/2105.14447.pdf)

#### 10.2. Overview
![](./model/img/psa.png)

#### 10.3. Usage Code
```python
from model.attention.PSA import PSA
import torch

input=torch.randn(50,512,7,7)
psa = PSA(channel=512,reduction=8)
output=psa(input)
print(output.shape)

```

***


### 11. Efficient Multi-Head Self-Attention Usage

#### 11.1. Paper
["ResT: An Efficient Transformer for Visual Recognition"](https://arxiv.org/abs/2105.13677)

#### 11.2. Overview
![](./model/img/EMSA.png)

#### 11.3. Usage Code
```python

from model.attention.EMSA import EMSA
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,64,512)
emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8,H=8,W=8,ratio=2,apply_transform=True)
output=emsa(input,input,input)
print(output.shape)
    
```

***


### 12. Shuffle Attention Usage

#### 12.1. Paper
["SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS"](https://arxiv.org/pdf/2102.00240.pdf)

#### 12.2. Overview
![](./model/img/ShuffleAttention.jpg)

#### 12.3. Usage Code
```python

from model.attention.ShuffleAttention import ShuffleAttention
import torch
from torch import nn
from torch.nn import functional as F


input=torch.randn(50,512,7,7)
se = ShuffleAttention(channel=512,G=8)
output=se(input)
print(output.shape)

    
```


***


### 13. MUSE Attention Usage

#### 13.1. Paper
["MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning"](https://arxiv.org/abs/1911.09483)

#### 13.2. Overview
![](./model/img/MUSE.png)

#### 13.3. Usage Code
```python
from model.attention.MUSEAttention import MUSEAttention
import torch
from torch import nn
from torch.nn import functional as F


input=torch.randn(50,49,512)
sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)

```

***


### 14. SGE Attention Usage

#### 14.1. Paper
[Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/pdf/1905.09646.pdf)

#### 14.2. Overview
![](./model/img/SGE.png)

#### 14.3. Usage Code
```python
from model.attention.SGE import SpatialGroupEnhance
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
sge = SpatialGroupEnhance(groups=8)
output=sge(input)
print(output.shape)

```

***


### 15. A2 Attention Usage

#### 15.1. Paper
[A2-Nets: Double Attention Networks](https://arxiv.org/pdf/1810.11579.pdf)

#### 15.2. Overview
![](./model/img/A2.png)

#### 15.3. Usage Code
```python
from model.attention.A2Atttention import DoubleAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
a2 = DoubleAttention(512,128,128,True)
output=a2(input)
print(output.shape)

```



### 16. AFT Attention Usage

#### 16.1. Paper
[An Attention Free Transformer](https://arxiv.org/pdf/2105.14103v1.pdf)

#### 16.2. Overview
![](./model/img/AFT.jpg)

#### 16.3. Usage Code
```python
from model.attention.AFT import AFT_FULL
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,49,512)
aft_full = AFT_FULL(d_model=512, n=49)
output=aft_full(input)
print(output.shape)

```



### 17. Outlook Attention Usage

#### 17.1. Paper


[VOLO: Vision Outlooker for Visual Recognition"](https://arxiv.org/abs/2106.13112)


#### 17.2. Overview
![](./model/img/OutlookAttention.png)

#### 17.3. Usage Code
```python
from model.attention.OutlookAttention import OutlookAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,28,28,512)
outlook = OutlookAttention(dim=512)
output=outlook(input)
print(output.shape)

```


***



### 18. ViP Attention Usage

#### 18.1. Paper


[Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition"](https://arxiv.org/abs/2106.12368)


#### 18.2. Overview
![](./model/img/ViP.png)

#### 18.3. Usage Code
```python

from model.attention.ViP import WeightedPermuteMLP
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(64,8,8,512)
seg_dim=8
vip=WeightedPermuteMLP(512,seg_dim)
out=vip(input)
print(out.shape)

```


***





### 19. CoAtNet Attention Usage

#### 19.1. Paper


[CoAtNet: Marrying Convolution and Attention for All Data Sizes"](https://arxiv.org/abs/2106.04803) 


#### 19.2. Overview
None


#### 19.3. Usage Code
```python

from model.attention.CoAtNet import CoAtNet
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,3,224,224)
mbconv=CoAtNet(in_ch=3,image_size=224)
out=mbconv(input)
print(out.shape)

```


***






### 20. HaloNet Attention Usage

#### 20.1. Paper


[Scaling Local Self-Attention for Parameter Efficient Visual Backbones"](https://arxiv.org/pdf/2103.12731.pdf) 


#### 20.2. Overview

![](./model/img/HaloNet.png)

#### 20.3. Usage Code
```python

from model.attention.HaloAttention import HaloAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,512,8,8)
halo = HaloAttention(dim=512,
    block_size=2,
    halo_size=1,)
output=halo(input)
print(output.shape)

```


***

### 21. Polarized Self-Attention Usage

#### 21.1. Paper

[Polarized Self-Attention: Towards High-quality Pixel-wise Regression"](https://arxiv.org/abs/2107.00782)  


#### 21.2. Overview

![](./model/img/PoSA.png)

#### 21.3. Usage Code
```python

from model.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,512,7,7)
psa = SequentialPolarizedSelfAttention(channel=512)
output=psa(input)
print(output.shape)


```


***


### 22. CoTAttention Usage

#### 22.1. Paper

[Contextual Transformer Networks for Visual Recognition---arXiv 2021.07.26](https://arxiv.org/abs/2107.12292) 


#### 22.2. Overview

![](./model/img/CoT.png)

#### 22.3. Usage Code
```python

from model.attention.CoTAttention import CoTAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
cot = CoTAttention(dim=512,kernel_size=3)
output=cot(input)
print(output.shape)



```

***


### 23. Residual Attention Usage

#### 23.1. Paper

[Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021](https://arxiv.org/abs/2108.02456) 


#### 23.2. Overview

![](./model/img/ResAtt.png)

#### 23.3. Usage Code
```python

from model.attention.ResidualAttention import ResidualAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
resatt = ResidualAttention(channel=512,num_class=1000,la=0.2)
output=resatt(input)
print(output.shape)



```

***



### 24. S2 Attention Usage

#### 24.1. Paper

[S²-MLPv2: Improved Spatial-Shift MLP Architecture for Vision---arXiv 2021.08.02](https://arxiv.org/abs/2108.01072) 


#### 24.2. Overview

![](./model/img/S2Attention.png)

#### 24.3. Usage Code
```python
from model.attention.S2Attention import S2Attention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
s2att = S2Attention(channels=512)
output=s2att(input)
print(output.shape)

```

***



### 25. GFNet Attention Usage

#### 25.1. Paper

[Global Filter Networks for Image Classification---arXiv 2021.07.01](https://arxiv.org/abs/2107.00645) 


#### 25.2. Overview

![](./model/img/GFNet.jpg)

#### 25.3. Usage Code - Implemented by [Wenliang Zhao (Author)](https://scholar.google.com/citations?user=lyPWvuEAAAAJ&hl=en)

```python
from model.attention.gfnet import GFNet
import torch
from torch import nn
from torch.nn import functional as F

x = torch.randn(1, 3, 224, 224)
gfnet = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=1000)
out = gfnet(x)
print(out.shape)

```

***


### 26. TripletAttention Usage

#### 26.1. Paper

[Rotate to Attend: Convolutional Triplet Attention Module---CVPR 2021](https://arxiv.org/abs/2010.03045) 

#### 26.2. Overview

![](./model/img/triplet.png)

#### 26.3. Usage Code - Implemented by [digantamisra98](https://github.com/digantamisra98)

```python
from model.attention.TripletAttention import TripletAttention
import torch
from torch import nn
from torch.nn import functional as F
input=torch.randn(50,512,7,7)
triplet = TripletAttention()
output=triplet(input)
print(output.shape)
```


***


### 27. Coordinate Attention Usage

#### 27.1. Paper

[Coordinate Attention for Efficient Mobile Network Design---CVPR 2021](https://arxiv.org/abs/2103.02907)


#### 27.2. Overview

![](./model/img/CoordAttention.png)

#### 27.3. Usage Code - Implemented by [Andrew-Qibin](https://github.com/Andrew-Qibin)

```python
from model.attention.CoordAttention import CoordAtt
import torch
from torch import nn
from torch.nn import functional as F

inp=torch.rand([2, 96, 56, 56])
inp_dim, oup_dim = 96, 96
reduction=32

coord_attention = CoordAtt(inp_dim, oup_dim, reduction=reduction)
output=coord_attention(inp)
print(output.shape)
```

***


### 28. MobileViT Attention Usage

#### 28.1. Paper

[MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2021.10.05](https://arxiv.org/abs/2103.02907)


#### 28.2. Overview

![](./model/img/MobileViTAttention.png)

#### 28.3. Usage Code

```python
from model.attention.MobileViTAttention import MobileViTAttention
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    m=MobileViTAttention()
    input=torch.randn(1,3,49,49)
    output=m(input)
    print(output.shape)  #output:(1,3,49,49)
    
```

***


### 29. ParNet Attention Usage

#### 29.1. Paper

[Non-deep Networks---ArXiv 2021.10.20](https://arxiv.org/abs/2110.07641)


#### 29.2. Overview

![](./model/img/ParNet.png)

#### 29.3. Usage Code

```python
from model.attention.ParNetAttention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    pna = ParNetAttention(channel=512)
    output=pna(input)
    print(output.shape) #50,512,7,7
    
```

***


### 30. UFO Attention Usage

#### 30.1. Paper

[UFO-ViT: High Performance Linear Vision Transformer without Softmax---ArXiv 2021.09.29](https://arxiv.org/abs/2110.07641)


#### 30.2. Overview

![](./model/img/UFO.png)

#### 30.3. Usage Code

```python
from model.attention.UFOAttention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    ufo = UFOAttention(d_model=512, d_k=512, d_v=512, h=8)
    output=ufo(input,input,input)
    print(output.shape) #[50, 49, 512]
    
```

-

### 31. ACmix Attention Usage

#### 31.1. Paper

[On the Integration of Self-Attention and Convolution](https://arxiv.org/pdf/2111.14556.pdf)

#### 31.2. Usage Code

```python
from model.attention.ACmix import ACmix
import torch

if __name__ == '__main__':
    input=torch.randn(50,256,7,7)
    acmix = ACmix(in_planes=256, out_planes=256)
    output=acmix(input)
    print(output.shape)
    
```

### 32. MobileViTv2 Attention Usage

#### 32.1. Paper

[Separable Self-attention for Mobile Vision Transformers---ArXiv 2022.06.06](https://arxiv.org/abs/2206.02680)


#### 32.2. Overview

![](./model/img/MobileViTv2.png)

#### 32.3. Usage Code

```python
from model.attention.MobileViTv2Attention import MobileViTv2Attention
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
    
```

### 33. DAT Attention Usage

#### 33.1. Paper

[Vision Transformer with Deformable Attention---CVPR2022](https://arxiv.org/abs/2201.00520)

#### 33.2. Usage Code

```python
from model.attention.DAT import DAT
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DAT(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        expansion=4,
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 7, 7] ,
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes=[False, False, False, False],
        strides=[-1, -1, 1, 1],
        sr_ratios=[-1, -1, -1, -1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
    )
    output=model(input)
    print(output[0].shape)
    
```

### 34. CrossFormer Attention Usage

#### 34.1. Paper

[CROSSFORMER: A VERSATILE VISION TRANSFORMER HINGING ON CROSS-SCALE ATTENTION---ICLR 2022](https://arxiv.org/pdf/2108.00154.pdf)

#### 34.2. Usage Code

```python
from model.attention.Crossformer import CrossFormer
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CrossFormer(img_size=224,
        patch_size=[4, 8, 16, 32],
        in_chans= 3,
        num_classes=1000,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=[7, 7, 7, 7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2,4], [2, 4]]
    )
    output=model(input)
    print(output.shape)
    
```

### 35. MOATransformer Attention Usage

#### 35.1. Paper

[Aggregating Global Features into Local Vision Transformer](https://arxiv.org/abs/2201.12903)

#### 35.2. Usage Code

```python
from model.attention.MOATransformer import MOATransformer
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = MOATransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6],
        num_heads=[3, 6, 12],
        window_size=14,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    output=model(input)
    print(output.shape)
    
```

### 36. CrissCrossAttention Attention Usage

#### 36.1. Paper

[CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721)

#### 36.2. Usage Code

```python
from model.attention.CrissCrossAttention import CrissCrossAttention
import torch

if __name__ == '__main__':
    input=torch.randn(3, 64, 7, 7)
    model = CrissCrossAttention(64)
    outputs = model(input)
    print(outputs.shape)
    
```

### 37. Axial_attention Attention Usage

#### 37.1. Paper

[Axial Attention in Multidimensional Transformers](https://arxiv.org/abs/1912.12180)

#### 37.2. Usage Code

```python
from model.attention.Axial_attention import AxialImageTransformer
import torch

if __name__ == '__main__':
    input=torch.randn(3, 128, 7, 7)
    model = AxialImageTransformer(
        dim = 128,
        depth = 12,
        reversible = True
    )
    outputs = model(input)
    print(outputs.shape)
    
```

***


# Backbone Series

- Pytorch implementation of ["Deep Residual Learning for Image Recognition---CVPR2016 Best Paper"](https://arxiv.org/pdf/1512.03385.pdf)

- Pytorch implementation of ["Aggregated Residual Transformations for Deep Neural Networks---CVPR2017"](https://arxiv.org/abs/1611.05431v2)

- Pytorch implementation of [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2020.10.05](https://arxiv.org/abs/2103.02907)

- Pytorch implementation of [Patches Are All You Need?---ICLR2022 (Under Review)](https://openreview.net/forum?id=TVHS5Y4dNvM)

- Pytorch implementation of [Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer---ArXiv 2021.06.07](https://arxiv.org/abs/2106.03650)

- Pytorch implementation of [ConTNet: Why not use convolution and transformer at the same time?---ArXiv 2021.04.27](https://arxiv.org/abs/2104.13497)

- Pytorch implementation of [Vision Transformers with Hierarchical Attention---ArXiv 2022.06.15](https://arxiv.org/abs/2106.03180)

- Pytorch implementation of [Co-Scale Conv-Attentional Image Transformers---ArXiv 2021.08.26](https://arxiv.org/abs/2104.06399)

- Pytorch implementation of [Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882)

- Pytorch implementation of [Rethinking Spatial Dimensions of Vision Transformers---ICCV 2021](https://arxiv.org/abs/2103.16302)

- Pytorch implementation of [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification---ICCV 2021](https://arxiv.org/abs/2103.14899)

- Pytorch implementation of [Transformer in Transformer---NeurIPS 2021](https://arxiv.org/abs/2103.00112)

- Pytorch implementation of [DeepViT: Towards Deeper Vision Transformer](https://arxiv.org/abs/2103.11886)

- Pytorch implementation of [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)
***

- Pytorch implementation of [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697)

- Pytorch implementation of [Augmenting Convolutional networks with attention-based aggregation](https://arxiv.org/abs/2112.13692)

- Pytorch implementation of [Going deeper with Image Transformers---ICCV 2021 (Oral)](https://arxiv.org/abs/2103.17239)

- Pytorch implementation of [Training data-efficient image transformers & distillation through attention---ICML 2021](https://arxiv.org/abs/2012.12877)

- Pytorch implementation of [LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/abs/2104.01136)

- Pytorch implementation of [VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/abs/2106.13112)

- Pytorch implementation of [Container: Context Aggregation Network---NeuIPS 2021](https://arxiv.org/abs/2106.01401)

- Pytorch implementation of [CMT: Convolutional Neural Networks Meet Vision Transformers---CVPR 2022](https://arxiv.org/abs/2107.06263)

- Pytorch implementation of [Vision Transformer with Deformable Attention---CVPR 2022](https://arxiv.org/abs/2201.00520)

- Pytorch implementation of [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

- Pytorch implementation of [ConvNeXtV2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)



