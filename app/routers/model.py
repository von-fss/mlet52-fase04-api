from fastapi import APIRouter
from pydantic import BaseModel
import torch
import torch.nn as nn

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {'description': 'Not Found'}}
)

@router.get('/train')
def train_model():
    return {"ok": 'ok'}

@router.post('/')
def value_model():
    return {{"ok": 'ok'}}