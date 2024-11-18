from fastapi.testclient import TestClient
import httpx
from app.main import app 

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'Hello Bigget Applications'}