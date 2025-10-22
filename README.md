# Machine-Teaching-for-XAI--TimeSeries-Models

## Project directory structure
Machine-Teaching-for-XAI--TimeSeries-Models </br>
├── **Data**  # dir for storing/caching all kinds of data </br>
├── **Figures**  # plots, images, etc. </br>
├── **Models**  # PyTorch models, training run checkpoints, etc. </br>
├── **Notebooks**  # For main project tasks such as model training, teaching set generation, and for testing and documenting code </br>
│   ├── 01__Data_Wrangling_and_FE.ipynb </br>
│   ├── 01__Data_Wrangling_and_FE.md </br>
│   ├── 02__EDA.ipynb </br>
│   ├── 02__EDA.md </br>
│   ├── 03__Modelling.ipynb </br>
│   ├── 03__Modelling.md </br>
│   ├── 04__Anomaly_Detection.ipynb </br>
│   ├── 04__Anomaly_Detection.md </br>
│   ├── 05__Curve_Simplification.ipynb </br>
│   ├── 05__Curve_Simplification.md </br>
│   ├── 06__Teach.ipynb </br>
│   └── notes.md </br>
├── **src** </br>
    ├── **api**  # backend api for MT4XAI web app  </br>
    ├── **app**  # frontend for MT4XAI web app </br>
    └── **mt4xai**  # convenent packaging of reusable project Python assets </br>
        ├── \_\_init\_\_.py </br>
        ├── data.py </br>
        ├── inference.py </br>
        ├── model.py </br>
        ├── ors.py </br>
        ├── plot.py </br>
        ├── teach.py </br>
        ├── test.py </br>
        └── training.py </br>
├── .env </br>
├── .gitignore </br>
├── LICENSE </br>
├── config.py </br>
├── config.yaml </br>
├── linux_requirements.txt</br>
└── pyproject.toml</br>