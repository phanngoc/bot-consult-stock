from altair import param
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import openai
import json

from dotenv import load_dotenv
import os

load_dotenv()

apiKey = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

print('apiKey', apiKey)

# Set up OpenAI API credentials
openai.api_key = apiKey

GPT_MODEL = "gpt-3.5-turbo-0613"
import requests

def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def execute_function_call(message):
    paramFn = json.loads(message["function_call"]["arguments"])
    results = globals()[message["function_call"]["name"]](paramFn)

    return results


def establish_context(system_message, user_message, functions=None):
    messages = []
    messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    chat_response = chat_completion_request(messages, functions)
    print('chat_response', chat_response.json())
    assistant_message = chat_response.json()["choices"][0]["message"]
    messages.append(assistant_message)
    if assistant_message.get("function_call"):
        results = execute_function_call(assistant_message)
        messages.append({"role": "function", "name": assistant_message["function_call"]["name"], "content": results})

    return messages

def generate_message(messages, user_message, functions=None):
    messages.append({"role": "user", "content": user_message})
    chat_response = chat_completion_request(messages, functions)
    assistant_message = chat_response.json()["choices"][0]["message"]
    messages.append(assistant_message)
    if assistant_message.get("function_call"):
        results = execute_function_call(assistant_message)
        messages.append({"role": "function", "name": assistant_message["function_call"]["name"], "content": results})

    return messages

functionsStock = [
    {
        "name": "forward_action",
        "description": "Lấy mã stock từ message",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_key": {
                    "type": "string",
                    "description": "Các mã stock của các công ty ở Việt Nam",
                },
                "action": {
                    "type": "string",
                    "enum": ["dự đoán", "so sánh", "báo cáo tài chính", "thông tin cơ bản"],
                    "description": "Lấy ra hành động mà người dùng mong muốn thực hiện",
                },
                "num_days": {
                    "type": "integer",
                    "description": "Số ngày dự đoán",
                }
            },
            "required": ["location", "format"],
        },
    },
]

from vnstock import *
from db import *

def fm_date(date):
    return date.strftime("%Y-%m-%d")

def fm_float(number):
    return "{:.2f}".format(number)

def fm_stock_code(code):
    return code.replace(" ", "")

def trim_words(sentence, num_word=200):
    # Split the sentence into words
    words = sentence.split()
    # Trim the sentence to 5 words
    trimmed_sentence = " ".join(words[:num_word])
    return trimmed_sentence

def compare_stock(stock_key):
    datals = stock_ls_analysis(stock_key, lang='vi')
    dataCompare = datals.to_string()
    # print('compare_stock', stock_key, dataCompare)
    prompt = "So sánh các mã " + stock_key + " dựa vào dữ liệu sau: \n" + dataCompare + ".\n"
    prompt = trim_words(prompt, 200)

    print('compare_stock prompt', prompt)
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        top_p=1.0,
        n=1,
        stop=None,
        temperature=0.3
    )

    res = response["choices"][0]["text"]
    print('compare_stock:res:', res)
    return res

from datetime import date, datetime, timedelta
import algorithm.rnn as stockRnn

# Get list of dates for the last 365 days
def get_last_days(num_days):
    current_date = datetime.now()
    days_ago = current_date - timedelta(days=num_days)
    date_list = [days_ago + timedelta(days=i) for i in range(num_days)]
    # Format the dates as "%Y-%m-%d" and store them in a new list
    formatted_dates = [fm_date(date) for date in date_list]

    return formatted_dates

import numpy as np

def get_predict_price(code, num_days=1):
    dates_train = get_last_days(365)
    df_his = stock_historical_data(code, dates_train[0], dates_train[len(dates_train) - 1], '1D', 'stock')
    data_train = df_his.iloc[:, 1:2].values
    rnn = stockRnn.TrainRnn()
    rnn.fit(data_train)
    # stock last 10 days for predict.
    date_train_test = df_his.iloc[:, 0:1].values
    date_train_test = date_train_test[len(date_train_test) - 10:].tolist()
    # Convert the 2D array back to the original list of datetime.date objects
    date_train_test = [date_array[0] for date_array in date_train_test]
    data_train_care = data_train[len(data_train) - 10:].tolist()
    
    response = []
    for k, date in enumerate(date_train_test):
        response.append({'date': fm_date(date), 'price': fm_float(data_train_care[k][0]), 'is_predict': False})
    
    data_test = [data_train_care]
    pred_result = rnn.predict(data_test)
    one_day = timedelta(days=1)
    new_day_predict = date_train_test[len(date_train_test)-1] + one_day
    date_train_test.append(new_day_predict)
    response.append({'date': fm_date(new_day_predict), 'price': fm_float(pred_result[0][0]), 'is_predict': True})

    while num_days > 1:
        data_test[0] = data_test[0][1:]
        data_test[0].append(pred_result[0])
        pred_result = rnn.predict(data_test)
        # add date predict.
        one_day = timedelta(days=1)
        new_day_predict = date_train_test[len(date_train_test)-1] + one_day
        date_train_test.append(new_day_predict)
        print('new_day_predict', date_train_test[len(date_train_test)-1], new_day_predict)
        response.append({'date': fm_date(new_day_predict), 'price': fm_float(pred_result[0][0]), 'is_predict': True})
        num_days = num_days - 1
    
    print('response:', response)
    return response

def predict_stock(args):
    print('predict_stock', args)
    stock_key = args['stock_key']
    num_days = args.get('num_days', 1)
    stock_codes = list(map(fm_stock_code, stock_key.split(",")))
    messages = []
    data = []
    print('stock_codes', stock_codes)
    for code in stock_codes:
        response_stock_price = get_predict_price(code, num_days)
        data.append({'code': code, 'data': response_stock_price})   
        predict_price_str = ','.join(str(e['price']) for e in response_stock_price if e['is_predict'])
        print('predict_price_str', predict_price_str)
        prompt = "Dự đoán giá cổ phiếu " + code + " là: " + predict_price_str + ".\n"
        messages.append(prompt)

    return {
        'status': 1,
        'type': 'du_doan',
        'message': ','.join(messages),
        'data': data,
        'isChart': True,
    }

def forward_action(params):
    action = params["action"]
    
    if action is not None:
        switcher = {
            "so sánh": compare_stock,
            "báo cáo tài chính": None,
            "thông tin cơ bản": None,
            "dự đoán": predict_stock
        }

        return switcher.get(action)(params)
    else:
        return None
    
# get_stock_key({"stock_key": "TPB", "action": "dự đoán", "num_days": 3})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/send-message", methods=["POST"])
def sendMessage():
    # Get the message from the POST request
    messageUser = request.json.get("prompt")

    # save message into database
    Message.create(user_id='test_user', role='user', message=messageUser)


    messages = establish_context("Trả lời câu hỏi của người dùng về tình trạng các mã stock ?",
        messageUser, functionsStock)
    # get last message from messages
    messageLast = messages[len(messages) - 1]
    content = messageLast['content']
    Message.create(user_id='test_user', role='function', message=content)
    content['isUser'] = False
    print('messageLast:content', content)

    return jsonify(content)

if __name__=='__main__':
    app.run(debug=True)

