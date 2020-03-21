class Config:
    def __init__(self, raw):
        self.compressorConfig = CompressorConfig(raw['compressorConfig'])
        self.imageHandlerConfig = ImageHandlerConfig(raw['imageHandlerConfig'])


class ImageHandlerConfig:
    def __init__(self, raw):
        self.imageLocation = raw['image_location']
        self.maxVolt = raw['maxVolt']


class CompressorConfig:
    def __init__(self, raw):
        self.curr = raw['curr']
        self.volt = raw['volt']


# config = Config(yaml.safe_load("""
# test1:
#     minVolt: -1
#     maxVolt: 1
# test2:
#     curr: 5
#     volt: 5
# """))
