from fastapi import APIRouter

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: { 'description': 'Not Found'}}
)

@router.get('/')
async def read_models():
    return [{ 'model': 'Predict Nvidia stock prices'}]