import requests
import pandas as pd
from pydantic import BaseModel
from dataclasses import dataclass, asdict

class Ticker(BaseModel):
    name: str 

@dataclass
class modelConfig:
    ticker: str = 'NVDA'
    time_step: int = 5
    epochs: int = 20
    optimizer: str = 'adam'
    batch_size: int = 15
    learning_rate: float = 0.05
    nn_activation: str = 'relu'
    nn_max_units: int = 100
    nn_layers: int = 2
    nn_return_sequences: bool = True
    loss: str = 'mean_squared_error'
    dropout: bool = True
    dropout_value: float = 0.2
    period: str = '3mo'

    def get_parameters(self) -> dict:
        return asdict(self)


def get_tickers_list() -> pd.DataFrame:
    """
    Description:
        Get all stock tickers from nasdaq and returns a dataframe.
    """
    nasdaq: str = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true'
    headers: str = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
    }

    session: requests = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get(nasdaq, timeout=10)
        response.raise_for_status()
        data: str = response.json()
        df: pd.DataFrame = pd.DataFrame(data.get("data", {}).get("rows", []))
        return df

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
    except requests.exceptions.Timeout as e:
        print(f"Timeout error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")