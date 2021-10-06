from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
from model_deploy.api.utils import ModelInput, ModelOutput, score_json
import click
from kedro.framework.cli import get_project_context
import uvicorn

# Overload from kedro catalog
regressor: LinearRegression
features: list

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


@click.group(name="API")
def commands():
    """Run fastapi"""

@commands.command()
def api():
    context = get_project_context()

    # Data Science params
    global regressor
    global features
    regressor = context.catalog.load("regressor")
    features = context.catalog.load("params:features")

    # App params
    host = context.catalog.load("params:host")
    port = context.catalog.load("params:port")

    uvicorn.run(app, host=host, port=port, log_level="info")