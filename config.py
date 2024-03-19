import json

class Config:
    def __init__(self, path) -> None:
        with open(path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            self.__dict__.update(data)
            
    @property
    def dict(self):
        return self.__dict__


if __name__ == "__main__":
    config = Config('./config.json')
    print(config.datasets)
    print(config.dict)