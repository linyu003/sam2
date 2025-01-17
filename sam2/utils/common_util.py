import numpy as np
import base64
import zlib
import gzip
import bz2 

compress_modules = {
    'bz2': bz2,
    'gzip': gzip,
    'zlib': zlib,
}


def serialize_ndarray(ndarray:np.ndarray,compress:str = 'bz2'):
    compress_module = compress_modules[compress]
    return {
        'data': base64.b64encode(compress_module.compress(ndarray.tobytes())).decode('utf-8'),
        'shape': list(ndarray.shape),
        'dtype': ndarray.dtype.name,
        'compress': compress,
    }


def parse_ndarray(data:dict):
    compress_module = compress_modules[data['compress']]
    return np.frombuffer(compress_module.decompress(base64.b64decode(data['data'])), dtype=data['dtype']).reshape(data['shape'])


def test_serialize_ndarray():
    # arr = np.random.rand(100, 100)
    arr = np.zeros((100,100))
    print(arr)
    serialized = serialize_ndarray(arr)
    print(serialized)
    deserialized = parse_ndarray(serialized)
    print(deserialized)
    assert np.allclose(arr, deserialized)

if __name__ == '__main__':
    test_serialize_ndarray()
