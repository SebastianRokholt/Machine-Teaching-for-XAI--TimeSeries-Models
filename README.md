## Project directory structure
Machine-Teaching-for-XAI--TimeSeries-Models
├── **Data**  # dir for storing/caching all kinds of data
├── **Figures**  # plots, images, etc.
├── **Models**  # PyTorch models, training run checkpoints, etc.
├── **Notebooks**  # For main project tasks such as model training, teaching set generation, and for testing and documenting code
│   ├── 01__Data_Wrangling_and_FE.ipynb
│   ├── 01__Data_Wrangling_and_FE.md
│   ├── 02__EDA.ipynb
│   ├── 02__EDA.md
│   ├── 03__Modelling.ipynb
│   ├── 03__Modelling.md
│   ├── 04__Anomaly_Detection.ipynb
│   ├── 04__Anomaly_Detection.md
│   ├── 05__Curve_Simplification.ipynb
│   ├── 05__Curve_Simplification.md
│   ├── 06__Teach.ipynb
│   └── notes.md
├── **src**
    ├── **api**  # backend api for MT4XAI web app 
    ├── **app**  # frontend fro MT4XAI web app
    └── **mt4xai**  # convenent packaging of reusable project Python assets
        ├── \_\_init\_\_.py
        ├── data.py
        ├── inference.py
        ├── model.py
        ├── ors.py
        ├── plot.py
        ├── teach.py
        ├── test.py
        └── training.py
├── .env
├── .gitignore
├── LICENSE
├── config.py
├── config.yaml
├── linux_requirements.txt
└── pyproject.toml