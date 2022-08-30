# -*- coding: utf-8 -*-
"""
@Time       : 2022/5/25 14:08
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : train 
@File       : FeiShuBot.py
@Software   : PyCharm
@Description: 
"""
import requests
import json


class FeiShuApp(object):
    def __init__(self, app_id=None, app_secret=None):
        super(FeiShuApp, self).__init__()
        self.feishu = FeiShu(app_id, app_secret)

    def send_message(self, receiver_phone: str, message: str):
        self.feishu.get_tenant_access_token()
        user_ids = self.feishu.get_batch_user_id_by_phones([receiver_phone])
        self.feishu.send_user_group_message(
            message,
            user_ids[0]
        )


class FeiShu:
    """
    飞书消息通知类
    """

    def __init__(self, app_id=None, app_secret=None, log=None):
        self.headers = None
        self.tenant_access_token = None
        self.BASE_URL = 'https://open.feishu.cn/open-apis'
        self.log = log
        self.app_id = app_id
        self.app_secret = app_secret

    def get_tenant_access_token(self):
        """
        根据邮箱获取用户ID
        """
        url = "/auth/v3/tenant_access_token/internal"
        req_body = json.dumps({
            "app_id": self.app_id,
            "app_secret": self.app_secret
        })
        data = self.get_request_data(url=url, data=req_body, method='post')
        self.tenant_access_token = data['tenant_access_token']
        #self.log.info("tenant_access_token: {0}".format(self.tenant_access_token))
        self.headers = {"Authorization": "Bearer {0}".format(self.tenant_access_token), "Content-Type": "application/json; charset=utf-8"}
        #self.log.info("headers: {0}".format(self.headers))

    def send_user_group_message(self, text, user_open_id, receive_id_type='open_id'):
        """发送用户消息"""
        url = "/im/v1/messages"
        req_body = json.dumps({
            "receive_id": user_open_id,
            'msg_type': 'text',
            'content': json.dumps({"text": text})
        })
        params = {"receive_id_type": receive_id_type}
        self.get_request_data(url=url, data=req_body, params=params, method='post')

    def get_batch_user_id_by_emails(self, emails):
        """
        根据邮箱获取用户ID
        """
        url = "/contact/v3/users/batch_get_id"
        req_body = json.dumps({
            "emails": emails
        })
        data = self.get_request_data(url=url, data=req_body, method='post')
        user_list = data['data']['user_list']
        user_ids = []
        for user in user_list:
            if 'user_id' in user:
                user_ids.append(user['user_id'])
        return user_ids

    def get_batch_user_id_by_phones(self, mobiles):
        """
        根据邮箱获取用户ID
        """
        url = "/contact/v3/users/batch_get_id"
        req_body = json.dumps({
            "mobiles": mobiles
        })
        data = self.get_request_data(url=url, data=req_body, method='post')
        user_list = data['data']['user_list']
        user_ids = []
        for user in user_list:
            if 'user_id' in user:
                user_ids.append(user['user_id'])
        return user_ids

    def get_group_id(self):
        """
        获取机器人所在的群信息
        """
        url = "/im/v1/chats"
        data = self.get_request_data(url=url)
        group_list = data['data']['items']
        group_ids = {}
        for group in group_list:
            if 'chat_id' in group:
                group_ids[group['chat_id']] = group['name']
        return group_ids

    def get_request_data(self, url=None, params=None, data=None, method='get'):
        """
        请求数据
        """
        url = "{0}/{1}".format(self.BASE_URL, url)
        if method == 'post':
            req = requests.post(url=url, data=data, params=params, headers=self.headers)
        elif method == 'get':
            req = requests.get(url=url, params=params, headers=self.headers)
        else:
            raise ValueError("暂不支持这种请求方式")
        if req.status_code == 200:
            return json.loads(req.text)
        else:
            raise ValueError("请求数据错误: {0}".format(req.text))