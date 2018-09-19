# 2018MathYouKnow

本项目为2018年研究生数学建模C题 "反恐分析与预测" 的代码部分

项目结构如下：
.
├── LICENSE                                     项目license  
├── README.md                                   项目说明  
├── data                                        存放原始数据和生成的数据  
│   ├── __init__.py  
│   ├── data_reader.py                          输入读入功能  
│   ├── dbscan\ final  
│   ├── gtd.pkl  
│   ├── gtd.xlsx                                原始文件  
│   ├── kmodes_1537119179.681623.csv            kmodes结果  
│   ├── kmodes_1537274480.292159.csv  
│   ├── kmodes_1537275309.722008.csv  
│   ├── kpro\ final  
│   ├── kprot_1537155581.9847062.csv  
│   ├── kprot_1537155975.758971.csv  
│   ├── kprot_1537158429.619887.csv  
│   ├── kprot_1537160313.2521348.csv  
│   ├── kprot_1537162524.993716.csv  
│   ├── kprot_1537274907.7650418.csv  
│   ├── kprot_centers_1537160313.2521348.csv  
│   ├── kprot_centers_1537162524.993716.csv  
│   ├── kprot_centers_1537274907.7650418.csv  
│   └── seg3  
├── env                                         virtualenv  
│   ├── bin  
│   ├── dist_metrics.pxd  
│   ├── include  
│   ├── lib  
│   └── pip-selfcheck.json  
├── requirements.txt  
├── seg1                                        第一题  
│   ├── DimensionValueError.py  
│   ├── __init__.py  
│   ├── pca.py  
│   ├── read_all.py  
│   ├── read_all_kprot.py  
│   ├── read_all_pca.py  
│   ├── test_all_funcs.py  
│   └── test_kmodes.py  
├── seg2                                        第二题  
│   ├── __init__.py  
│   ├── dbscan_gen_res.py  
│   ├── hdbscan_gen_res.py  
│   └── rf_with_dim_reduce.py  
└── seg3                                        第三题  
    ├── __init__.py  
    ├── map.py  
    ├── pyechart.py  
    ├── render.html  
    └── spatiotemporal_gen.py  
