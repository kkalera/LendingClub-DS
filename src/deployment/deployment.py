import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import os


app = FastAPI()


class Data(BaseModel):
    instances: dict


class Instance(BaseModel):
    instances: list[dict]


def make_prediction(instances):
    df = pd.DataFrame.from_records(instances, index=[0])

    # model_0 = joblib.load(os.getenv("MODEL_0"))
    model_0 = joblib.load("models/anomaly_pipe.pkl")
    df["anomaly"] = model_0.predict(
        df[
            [
                "total_credit",
                "open_credit",
                "AMT_ANNUITY",
                "AMT_CREDIT",
                "DAYS_EMPLOYED",
                "AMT_INCOME_TOTAL",
                "DAYS_BIRTH",
                "DAYS_ID_PUBLISH",
                "AMT_GOODS_PRICE",
                "closed_credit",
            ]
        ]
    )

    # model_1 = joblib.load(os.getenv("MODEL_1"))
    model_1 = joblib.load("models/segmentation_pipe.pkl")
    df["cluster"] = model_1.predict(df)

    # model_2 = joblib.load(os.getenv("MODEL_2"))
    model_2 = joblib.load("models/xgb_credit_approval_2.pkl")
    df["prediction"] = model_2.predict(
        df[
            [
                "NAME_CONTRACT_TYPE",
                "CODE_GENDER",
                "FLAG_OWN_CAR",
                "FLAG_OWN_REALTY",
                "CNT_CHILDREN",
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "AMT_GOODS_PRICE",
                "NAME_TYPE_SUITE",
                "NAME_INCOME_TYPE",
                "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS",
                "NAME_HOUSING_TYPE",
                "REGION_POPULATION_RELATIVE",
                "DAYS_BIRTH",
                "DAYS_EMPLOYED",
                "DAYS_REGISTRATION",
                "DAYS_ID_PUBLISH",
                "OWN_CAR_AGE",
                "FLAG_MOBIL",
                "FLAG_EMP_PHONE",
                "FLAG_WORK_PHONE",
                "FLAG_CONT_MOBILE",
                "FLAG_PHONE",
                "FLAG_EMAIL",
                "OCCUPATION_TYPE",
                "CNT_FAM_MEMBERS",
                "REGION_RATING_CLIENT",
                "REGION_RATING_CLIENT_W_CITY",
                "WEEKDAY_APPR_PROCESS_START",
                "HOUR_APPR_PROCESS_START",
                "REG_REGION_NOT_LIVE_REGION",
                "REG_REGION_NOT_WORK_REGION",
                "LIVE_REGION_NOT_WORK_REGION",
                "REG_CITY_NOT_LIVE_CITY",
                "REG_CITY_NOT_WORK_CITY",
                "LIVE_CITY_NOT_WORK_CITY",
                "ORGANIZATION_TYPE",
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
                "APARTMENTS_AVG",
                "BASEMENTAREA_AVG",
                "YEARS_BEGINEXPLUATATION_AVG",
                "YEARS_BUILD_AVG",
                "COMMONAREA_AVG",
                "ELEVATORS_AVG",
                "ENTRANCES_AVG",
                "FLOORSMAX_AVG",
                "FLOORSMIN_AVG",
                "LANDAREA_AVG",
                "LIVINGAPARTMENTS_AVG",
                "NONLIVINGAREA_AVG",
                "APARTMENTS_MODE",
                "BASEMENTAREA_MODE",
                "YEARS_BEGINEXPLUATATION_MODE",
                "YEARS_BUILD_MODE",
                "ENTRANCES_MODE",
                "LANDAREA_MODE",
                "LIVINGAPARTMENTS_MODE",
                "NONLIVINGAPARTMENTS_MODE",
                "NONLIVINGAREA_MODE",
                "APARTMENTS_MEDI",
                "BASEMENTAREA_MEDI",
                "YEARS_BEGINEXPLUATATION_MEDI",
                "YEARS_BUILD_MEDI",
                "COMMONAREA_MEDI",
                "LANDAREA_MEDI",
                "LIVINGAREA_MEDI",
                "TOTALAREA_MODE",
                "OBS_30_CNT_SOCIAL_CIRCLE",
                "OBS_60_CNT_SOCIAL_CIRCLE",
                "DAYS_LAST_PHONE_CHANGE",
                "FLAG_DOCUMENT_3",
                "FLAG_DOCUMENT_16",
                "FLAG_DOCUMENT_19",
                "FLAG_DOCUMENT_20",
                "FLAG_DOCUMENT_21",
                "AMT_REQ_CREDIT_BUREAU_DAY",
                "AMT_REQ_CREDIT_BUREAU_WEEK",
                "AMT_REQ_CREDIT_BUREAU_MON",
                "AMT_REQ_CREDIT_BUREAU_QRT",
                "AMT_REQ_CREDIT_BUREAU_YEAR",
                "closed_loans",
                "active_loans",
                "open_credit",
                "closed_credit",
                "total_credit",
                "dti",
                "cluster",
            ]
        ]
    )
    anomaly = True if df["anomaly"].values[0] == -1 else False
    issues = True if df["prediction"].values[0] == 1 else False
    return [anomaly, int(df["cluster"].values[0]), issues]


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    print("----------------- Predicting -----------------")
    pred = make_prediction(body["instances"])
    print(pred)
    print("----------------- Done Predicting -----------------")
    return {"predictions": [{"anomaly": pred[0], "cluster": pred[1], "issues": pred[2]}]}


@app.get("/health")
def health():
    return 200


@app.get("/")
def root():
    return {"message": "Welcome to the API!"}
