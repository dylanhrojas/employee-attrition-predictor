from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_valid_pred(valid_employee_payload):
    response = client.post("/predict", json=valid_employee_payload)
    assert response.status_code == 200

def test_out_of_range(valid_employee_payload):
    payload = {**valid_employee_payload}
    payload["StockOptionLevel"] = 4

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_missing_field(valid_employee_payload):
    payload = {**valid_employee_payload}
    del payload["StockOptionLevel"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_default_values(valid_employee_payload):
    payload = {**valid_employee_payload}
    del payload["EmployeeCount"]
    del payload["StandardHours"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200