import os, json
from typing import Sequence
import config_data as config
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage


def _window(messages: list[BaseMessage], limit: int) -> list[BaseMessage]:
    """只保留最近 limit 条，并保证以用户消息开头（不切断 user/assistant 配对）。"""
    if limit <= 0:
        return []
    recent = messages[-limit:]
    # 正常历史长度恒为偶数，尾部切片天然以 human 开头；
    # 这里兜底脏数据，避免把半轮对话塞给模型。
    while recent and recent[0].type != "human":
        recent = recent[1:]
    return recent


def get_history(session_id):
    return FileChatMessageHistory(session_id=session_id, storage_path="./chat_histories")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id         #会话id
        self.storage_path = storage_path     #不同会话id的存储文件，所在的文件夹路径
        
        #完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)  

        #确保文件夹是存在的，如果不存在就创建
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        #Sequence序列 类似list
        
        # 必须基于「全量」追加，不能走 self.messages（那是带窗口的读视图），
        # 否则窗口截断会连带把磁盘上的老历史永久删掉。
        all_messages = list(self._load_all_messages())
        all_messages.extend(messages)         #把新消息添加到已有的消息列表中

        # 磁盘侧也限长：只保留最近 N 条；若配置得比 prompt 窗口还小则按窗口兜底。
        disk_limit = max(int(config.history_disk_max_messages), int(config.history_max_messages))
        all_messages = _window(all_messages, disk_limit)

        # 将数据同步写入本地文件中
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串形式写入）
        # 官方message_to_dict: BaseMessage对象 -> dict
        new_messages = []
        for msg in all_messages:
            d = message_to_dict(msg)   #BaseMessage对象 -> dict
            new_messages.append(d)
        
        #等效写法，更简洁
        # new_messages = [message_to_dict(msg) for msg in all_messages]

        # 将数据写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)
      
    def _load_all_messages(self) -> list[BaseMessage]:
        """读取文件里的全量历史，不做任何截断（供窗口读与追加写共用）。"""
        # 当前文件内：list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)   #返回值list[字典]
                return messages_from_dict(messages_data)  #list[字典] -> list[BaseMessage对象]
        except FileNotFoundError:
            return []

    @property       #装饰器，把一个方法变成属性调用
    def messages(self) -> list[BaseMessage]:
        """交给 LLM 的历史窗口：只保留最近 history_max_messages 条。"""
        return _window(self._load_all_messages(), int(config.history_max_messages))

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)   #清空文件内容，写入一个空的列表