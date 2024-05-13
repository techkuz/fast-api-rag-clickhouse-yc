import requests
import json

FAST_API_HOST = 'http://127.0.0.1:8000'
CH_HOST = ''
CH_PORT = '8443'
CH_USER = 'admin'
CH_PASSWORD = ''
CA_CERT = '.clickhouse/root.crt'
FOLDER_ID = ''
S3_PREFIX = ''
S3_BUCKET = ''
S3_URL = 'https://storage.yandexcloud.net'
S3_ACCESS_KEY_ID = ''
S3_SECRET_ACCESS_KEY = ''
TOKEN = ''


def get_clickhouse_status():

    url = f"{FAST_API_HOST}/clickhouse_connection?host={CH_HOST}&port={PORT}&user={CH_USER}&password={CH_PASSWORD}&ca_cert={CA_CERT}"

    payload = {}
    headers = {
        'Cache-Control': 'no-cache',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip,deflate',
        'Connection': 'keep-alive'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)


def post_embeddings():

    url = f"{FAST_API_HOST}/embeddings?ch_host={CH_HOST}&ch_port={CH_PORT}&ch_user={CH_USER}&ch_password={CH_PASSWORD}&ch_ca_cert={CA_CERT}&folder_id={FOLDER_ID}&s3_bucket={S3_BUCKET}&s3_prefix={S3_PREFIX}&s3_url={S3_URL}/&token={TOKEN}&s3_access_key_id={S3_ACCESS_KEY_ID}&s3_secret_access_key={S3_SECRET_ACCESS_KEY}"

    payload = {}
    headers = {
        'Cache-Control': 'no-cache',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip,deflate',
        'Connection': 'keep-alive'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


def similar_docs_search():
    url = f"{FAST_API_HOST}/similar_docs"

    payload = json.dumps({
        "text": "Могу ли я использовать эти API в pytorch 2.3.0?\n# torch.jit.quantized APIs\ntorch.jit.quantized.quantize_rnn_cell_modules\ntorch.jit.quantized.quantize_rnn_modules\ntorch.jit.quantized.quantize_linear_modules\ntorch.jit.quantized.QuantizedLinear\ntorch.jit.QuantizedLinearFP16\ntorch.jit.quantized.QuantizedGRU\ntorch.jit.quantized.QuantizedGRUCell\ntorch.jit.quantized.QuantizedLSTM\ntorch.jit.quantized.QuantizedLSTMCell"
    })
    headers = {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip,deflate',
        'Connection': 'keep-alive'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)


def test_emb_docs():
    url = f"{FAST_API_HOST}/emb_docs"

    payload = {}
    headers = {
        'Cache-Control': 'no-cache',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip,deflate',
        'Connection': 'keep-alive'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)


def query_llm():

    url = f"{FAST_API_HOST}/llm?folder_id={FOLDER_ID}&token={TOKEN}"

    payload = json.dumps({
        "text": "Могу ли я использовать эти API в pytorch 2.3.0?\n# torch.jit.quantized APIs\ntorch.jit.quantized.quantize_rnn_cell_modules\ntorch.jit.quantized.quantize_rnn_modules\ntorch.jit.quantized.quantize_linear_modules\ntorch.jit.quantized.QuantizedLinear\ntorch.jit.QuantizedLinearFP16\ntorch.jit.quantized.QuantizedGRU\ntorch.jit.quantized.QuantizedGRUCell\ntorch.jit.quantized.QuantizedLSTM\ntorch.jit.quantized.QuantizedLSTMCell"
    })
    headers = {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip,deflate',
        'Connection': 'keep-alive'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


def query_llm_rag():

    url = f"{FAST_API_HOST}/llm_rag?folder_id={FOLDER_ID}&token={TOKEN}"

    payload = json.dumps({
        "text": "Могу ли я использовать эти API в pytorch 2.3.0?\n# torch.jit.quantized APIs\ntorch.jit.quantized.quantize_rnn_cell_modules\ntorch.jit.quantized.quantize_rnn_modules\ntorch.jit.quantized.quantize_linear_modules\ntorch.jit.quantized.QuantizedLinear\ntorch.jit.QuantizedLinearFP16\ntorch.jit.quantized.QuantizedGRU\ntorch.jit.quantized.QuantizedGRUCell\ntorch.jit.quantized.QuantizedLSTM\ntorch.jit.quantized.QuantizedLSTMCell"
    })
    headers = {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip,deflate',
        'Connection': 'keep-alive'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
