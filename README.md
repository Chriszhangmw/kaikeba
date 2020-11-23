

Text Summary
===========================
Text Summary

### Usage:
1. Cleaning and generating training data:
 run data_loader.py
2. Training, modify the model in main.py：
![1](https://raw.github.com/Chriszhangmw/kaikeba/master/code/result/1.png)
3. Testing, modify the model in main.py:
![2](https://raw.github.com/Chriszhangmw/kaikeba/master/code/result/3.png)
![3](https://raw.github.com/Chriszhangmw/kaikeba/master/code/result/2.png)

# 会话平台接口
[toc]
## 功能性接口
### 机器人
#### 机器人管理
* 新增技能：
```json
PUT /user/some_user_id
请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```

* 修改技能：

```json
PUT /user/some_user_id

请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```
#### 机器人列表
* 新增技能：
```json
PUT /user/some_user_id
请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```

* 修改技能：

```json
PUT /user/some_user_id

请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```
#### 创建机器人
* 新增技能：
```json
PUT /user/some_user_id
请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```

* 修改技能：

```json
PUT /user/some_user_id

请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```
### 我的技能
#### 技能管理
* 新增技能：接口的描述，该接口主要作用是新增撒

```json
PUT /user/some_user_id
请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```

* 修改技能：

```json
PUT /user/some_user_id

请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```
#### FAQ技能
* 新增技能：
```json
PUT /user/some_user_id
请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```

* 修改技能：

```json
PUT /user/some_user_id

请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```
#### 导航管理
* 新增技能：
```json
PUT /user/some_user_id
请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```

* 修改技能：

```json
PUT /user/some_user_id

请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```
#### 任务型技能
* 新增技能：
```json
PUT /user/some_user_id
请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```

* 修改技能：

```json
PUT /user/some_user_id

请求参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
响应参数：
{
  "name": "zhangsan",
  "email": "yyy@y.com"
}
添加时间：{2020-11-17}
```
### 我的NLU
#### 分词
#### NLU
#### 意图理解
#### 实体识别
#### 实体对齐
#### 搜索

### 我的知识
#### 问答
#### 图数据
#### 文档
#### 表格
#### 关键词
#### 同义词
#### 问答库
#### 答复
#### 模型服务
#### FAQ问题管理
#### KBQA实体查询

### 文档中心
#### 文档中心1
#### 文档中心2
## 非功能性接口
### 个人管理
### 登录/登出管理
