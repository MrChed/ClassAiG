import telebot
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

bot = telebot.TeleBot('Api-key')
model = ResNet50(weights='imagenet')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Получение фотографии из сообщения
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_url = f'https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}'
    response = requests.get(file_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Классификация изображения
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Формирование сообщения с результатами классификации
    message_text = 'Результаты классификации:\n'
    for pred in decoded_preds:
        message_text += f'{pred[1]} с вероятностью {pred[2]:.2%}\n'

    bot.reply_to(message, message_text)

bot.polling()
