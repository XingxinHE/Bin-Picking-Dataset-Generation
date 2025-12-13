
import pytest

def pytest_addoption(parser):
    parser.addoption("--dataset_name", action="store", default="teris", help="Name of the dataset to test (e.g. teris)")
    parser.addoption("--model_name", action="store", default="teris", help="Name of the model folder (e.g. teris)")

@pytest.fixture(scope="session")
def dataset_name(request):
    return request.config.getoption("--dataset_name")

@pytest.fixture(scope="session")
def model_name(request):
    return request.config.getoption("--model_name")
