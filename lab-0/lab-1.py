import base64
import io
import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()


def pil_to_base64(image: Image.Image) -> str:
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


SYSTEM_PROMPT = """
Ты - OCR агент по распознаванию текста на изображении.
На вход ты получаешь скриншот отсканированной страницы книги.
На выходе требуется предоставть форматированный в Markdown редактируемый текст. 
Характер содержимого на скриншотах - научно-технический. 
В нем вместе с текстом могут соедержаться таблицы и формулы. 
Форматируй согласно правилам разметки Markdown. 
Строго сохраняй исходный текст, не вноси ничего нового и не сокращай. 
Старайся приблизить исходное форматирование текста
"""

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def process_image(image):
    base64_image = pil_to_base64(image)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.strip(),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )

        result = chat_completion.choices[0].message.content
        return result
    except Exception as e:
        return f"Ошибка при обработке: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("Распознавание текста на изображении (OCR) через Groq api")
    image_input = gr.Image(type="pil", label="Загрузите изображение")
    output_md = gr.Markdown(
        label="Результат",
        latex_delimiters=[
            {
                "left": "$$",
                "right": "$$",
                "display": True,
            },
            {"left": "$", "right": "$", "display": False},
        ],
    )
    btn = gr.Button("Распознать текст")

    btn.click(fn=process_image, inputs=image_input, outputs=output_md)

demo.launch()
