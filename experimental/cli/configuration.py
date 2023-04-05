import os
import configparser
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, ServiceContext, LLMPredictor
from gpt_index.data_structs.data_structs_v2 import SimpleIndexDict


CONFIG_FILE_NAME = 'config.ini'
DEFAULT_CONFIG = {
    "index": {
        "type": "json"
    },
    "embed_model": {
        "type": "default"
    },
    "llm_predictor": {
        "type": "default"
    }
}


def load_config(root="."):
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.read(os.path.join(root, CONFIG_FILE_NAME))
    return config


def save_config(config, root="."):
    with open(os.path.join(root, CONFIG_FILE_NAME), 'w') as fd:
        config.write(fd)


def load_index(root="."):
    config = load_config(root)
    service_context = _load_service_context(config)
    if config["index"]["type"] == "json":
        index_file = os.path.join(root, 'index.json')
    else:
        raise KeyError(f"Unknown index.type {config['index']['type']}")
    if os.path.exists(index_file):
        return GPTSimpleVectorIndex.load_from_disk(index_file, service_context=service_context)
    else:
        return GPTSimpleVectorIndex(index_struct=SimpleIndexDict(), service_context=service_context)


def save_index(index, root="."):
    config = load_config(root)
    if config["index"]["type"] == "json":
        index_file = os.path.join(root, 'index.json')
    else:
        raise KeyError(f"Unknown index.type {config['index']['type']}")
    index.save_to_disk(index_file)


def _load_service_context(config):
    embed_model = _load_embed_model(config)
    llm_predictor = _load_llm_predictor(config)
    return ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)


def _load_llm_predictor(config):
    model_type = config["llm_predictor"]["type"].lower()
    if model_type == "default":
        return LLMPredictor()
    if model_type == "azure":
        engine = config["llm_predictor"]["engine"]
        return LLMPredictor(llm=OpenAI(engine=engine))
    else:
        raise KeyError("llm_predictor.type")


def _load_embed_model(config):
    model_type = config["embed_model"]["type"]
    if model_type == "default":
        return OpenAIEmbedding()
    else:
        raise KeyError("embed_model.type")
