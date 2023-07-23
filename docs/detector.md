



```
Model summary: 322 layers, 86254162 parameters, 0 gradients, 204.0 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 1523/1523 [17:15<00:00,  1.47it/s]
                   all      48736      48736      0.938      0.907      0.951      0.828
               Insecta      48736      16732      0.984      0.963      0.989      0.892
                  Aves      48736      17314      0.985      0.966      0.991      0.858
              Reptilia      48736       5480      0.956       0.92      0.967      0.836
              Mammalia      48736       2654      0.934      0.884       0.95      0.768
              Amphibia      48736       2280      0.953      0.954      0.974      0.874
              Mollusca      48736       1571      0.901      0.876      0.927      0.834
              Animalia      48736       1143      0.874      0.822      0.879      0.718
             Arachnida      48736       1051      0.913      0.934      0.965      0.856
        Actinopterygii      48736        511      0.941      0.845      0.915      0.814
```





## Testing Pipeline

This is one sample of the folders for testing data.

```
test/
├── 01432_Animalia_Arthropoda_Insecta_Lepidoptera_Limacodidae_Isochaetes_beutenmuelleri
│   ├── de471378-9e0c-4d24-87a6-b58851382d95.jpg
│   └── f0d665bd-1138-4ae8-a727-6f88f184e942.jpg
├── 01433_Animalia_Arthropoda_Insecta_Lepidoptera_Limacodidae_Lithacodes_fasciola
│   ├── e128c91a-e341-4106-96dc-80f4ec875de3.jpg
│   └── f81a4c65-de9f-482a-a6e1-bbe97f72c780.jpg
└── 02698_Animalia_Arthropoda_Insecta_Zygentoma_Lepismatidae_Ctenolepisma_lineata
    ├── d374da7b-e2f8-4138-b75c-de43a2aea18b.jpg
    └── e74967d1-ad4b-4dc0-92ad-07dcd4f573fb.jpg
```



Load one image, detect firstly to get the bounding box or ROI image by the detection model., then send this detection result to the classification model and get the classification result.

The ground truth can be extracted from the folder name of the testing image.

Finally, we will get the accuracy of top 1-5 as the testing result (`result.txt`), and failure cases will be saved in file to track the image path (`failure.txt)`.



### Notice

One image may be classified by 1-2 models to generate 1-2 results, then the final result will be merged by all of them.

For Insecta detection, in this stage, we will prepare 6 classification models for each group of Insecta, and we will have 6 testing pipelines for each of them. So, it is "detection+classification", which is a little different from Production Inference Pipeline: "detection+classification+classification". Now we only focus on the former one.
