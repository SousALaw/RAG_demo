import os, json
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage

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
        
        all_messages = list(self.messages)    #已有的消息列表
        all_messages.extend(messages)         #把新消息添加到已有的消息列表中

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
      
    @property       #装饰器，把一个方法变成属性调用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内：list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)   #返回值list[字典]
                return messages_from_dict(messages_data)  #list[字典] -> list[BaseMessage对象]
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)   #清空文件内容，写入一个空的列表