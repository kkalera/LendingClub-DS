# Retail Bank Risk Evaluation

&nbsp;


**Context**

This project uses the home credit default risk dataset that can be found on [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) to create 3 products that aim to add value to retail banks.

- The first product is an anomaly detection model to detect anomalous applications.

    **Results:** *The model flags about 8% of the applications as anomalous. The anomalous applications have a larger ratio of repayment issues. Thus being of value for subsequent models.*

&nbsp;

- The second product is a segmentation model that segments the application into one of 5 clusters. This model also uses the flag from the anomaly detection model.

    **Results:** *The model segments the applications into 5 different clusters. There is a significant difference between the repayment issues between different clusters. Again, providing value for subsequent models and later processes like setting interest rates.*

&nbsp;

- The third and final product is a classification model that classify's whether or not an application is likely to experience repayment issues.
    
    **Results:** *The model is able to provide as much value to the bank as the status quo. Using the model should thus not result in lost revenue while enabling the automation of the entire approval process. This can increase productivity of the people that are involved in this process and thus increasing revenue.*

&nbsp;

&nbsp;

**Project Structure**
```
.
└── Project Home/
    ├── src/
    │   ├── data/
    │   │   ├── application_test.csv
    │   │   ├── application_train.csv
    │   │   ├── bureau_balance.csv
    │   │   ├── bureau.csv
    │   │   ├── credit_card_balance.csv
    │   │   ├── HomeCredit_columns_description.csv
    │   │   ├── installments_payments.csv
    │   │   ├── POS_CASH_balance.csv
    │   │   └── previous_application.csv
    │   ├── deployment/
    │   │   ├── models/
    │   │   │   ├── anomaly_pipe.pkl
    │   │   │   ├── segmentation_pipe.pkl
    │   │   │   └── xgb_credit_approval_2.pkl
    │   │   ├── src/
    │   │   │   └── lib/
    │   │   │       ├── data_functions.py
    │   │   │       └── ml_functions.py
    │   │   ├── deployment_commands.txt
    │   │   ├── deployment.py
    │   │   ├── Dockerfile
    │   │   └── requirements.txt
    │   ├── img/
    │   │   └── data_diagram.jpg
    │   ├── lib/
    │   │   ├── data_functions.py
    │   │   ├── ml_functions.py
    │   │   └── plot_functions.py
    │   ├── logs/
    │   │   └── my.log
    │   ├── models/
    │   │   ├── anomaly_pipe.pkl
    │   │   ├── rfecv.pkl
    │   │   ├── segmentation_pipe.pkl
    │   │   ├── xgb_credit_approval_0.pkl
    │   │   ├── xgb_credit_approval_1.pkl
    │   │   └── xgb_credit_approval_2.pkl
    │   ├── request_sample.json
    │   └── requirements.txt
    ├── .gitignore
    ├── 341.ipynb
    ├── Project.ipynb
    └── readme.md
```

&nbsp;

&nbsp;

## Local Execution

If you'd like to run this notebook locally, you can download the dataset using [this link](https://storage.googleapis.com/341-home-credit-default/home-credit-default-risk.zip) and extract it into the ```src/data``` folder. After that you can run the notebook up until the deployment part since this requires an authentication code to access the models on google cloud.
