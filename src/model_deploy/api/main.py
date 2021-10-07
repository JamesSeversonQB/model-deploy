from fastapi import FastAPI

from model_deploy.api.utils import (
    ModelInput,
    ModelOutput,
    get_features,
    get_regressor,
    score_json,
)

regressor = get_regressor()
features = get_features()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to your model deployment via api!"}


@app.post("/score/", response_model=ModelOutput)
def score_one(model_input: ModelInput) -> dict:
    uuid = model_input.uuid
    data = model_input.dict()
    score = score_json(regressor, data, features)

    response = dict(uuid=uuid, score=score)
    return response
