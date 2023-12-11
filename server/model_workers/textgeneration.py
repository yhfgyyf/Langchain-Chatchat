import sys
from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from fastchat import conversation as conv
import json
from typing import List, Dict
from configs import logger, log_verbose


class TextGenerationWorker(ApiModelWorker):
    def __init__(
            self,
            *,
            controller_addr: str = None,
            worker_addr: str = None,
            model_names: List[str] = ["textgeneration-api"],
            **kwargs,
    ):
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 4096) 
        super().__init__(**kwargs)


    def do_chat(self, params: ApiChatParams) -> Dict:
        params.load_config(self.model_names[0])
        data = dict(
            model=params.deployment_name,   #Llama factory项目需要填写模型名称
            messages=params.messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            stream=True,
        )
        url = ("{}/chat/completions".format(params.api_base_url))
        headers = {
            'Content-Type': 'application/json',
            'api-key': params.api_key,
        }

        text = ""
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:url: {url}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')
            logger.info(f'{self.__class__.__name__}:data: {data}')

        with get_httpx_client() as client:
            with client.stream("POST", url, headers=headers, json=data) as response:
                for line in response.iter_lines():
                    if not line.strip() or "[DONE]" in line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    resp = json.loads(line)
                    if choices := resp["choices"]:
                        delta = choices[0].get("delta", {})
                        if 'content' in delta:  # 检查是否存在content字段,兼容Llama-factory的openai接口
                            if chunk := delta.get("content"):
                                text += chunk
                                yield {
                                    "error_code": 0,
                                    "text": text
                                }
                        else:
                            pass
                    else:
                        self.logger.error(f"请求API时发生错误：{resp}")

    def get_embeddings(self, params):
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="You are a helpful, respectful and honest assistant.",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )


if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.base_model_worker import app

    worker = TextGenerationWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21010",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21010)
