# ELHRN

Code relaese for [Enhanced Local Homogenization and Reconstruction Network for Fine-grained Few-shot Image Classification]

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yaml
  conda activate ELHRN
  ```

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN), but the categories  of train, val, and test follows split.txt. And then move the processed dataset  to directory ./data.

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- cars \[[Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)\]
- dogs \[[Download Link](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link)\]
- tiered-ImageNet \[[Dataset Page](https://github.com/renmengye/few-shot-ssl-public), [Download Link](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view)\]
- tiered-ImageNet_DeepEMD (derived from [DeepEMD](https://arxiv.org/abs/2003.06777)'s [implementation](https://github.com/icoz69/DeepEMD)) \[[Dataset Page](https://github.com/icoz69/DeepEMD), [Download Link](https://drive.google.com/file/d/1ANczVwnI1BDHIF65TgulaGALFnXBvRfs/view)\]
- iNaturalist2017 \[[Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017), [Download Data](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz), [Download Annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip)\]

## Train

* To train ELHRN on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/ELHRN/Conv-4
  ./train_first.sh
  ```

  ```shell
  cd experiments/CUB_fewshot_cropped/ELHRN/Conv-4
  ./train_second.sh
  ```
  
* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/ELHRN/ResNet-12
  ./train_first.sh
  ```
  
  ```shell
  cd experiments/CUB_fewshot_cropped/ELHRN/ResNet-12
  ./train_second.sh
  ```
  

## Train-Snapshot

* To train ELHRN on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/ELHRN_Snapshot/Conv-4
  ./train_first.sh
  ```

* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/ELHRN_Snapshot/ResNet-12
  ./train_second.sh
  ```


## Test

```shell
    cd experiments/CUB_fewshot_cropped/ELHRN/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/ELHRN/ResNet-12
    python ./test.py
```

## Test-Snapshot

```shell
    cd experiments/CUB_fewshot_cropped/ELHRN_Snapshot/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/ELHRN_Snapshot/ResNet-12
    python ./test.py
```

## References

Thanks to  [Davis](https://github.com/Tsingularity/FRN), [Phil](https://github.com/lucidrains/vit-pytorch) and  [Yassine](https://github.com/yassouali/SCL), for the preliminary implementations.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- meiyinhu@ncu.edu.cn
