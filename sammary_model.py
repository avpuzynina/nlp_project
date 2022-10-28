import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def init_model():
    global tokenizer
    global model
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# # инициализация модели: архитектура + веса
def samm_model(text):
    input_ids = tokenizer(
                            [text],
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=512
                        )["input_ids"]

    output_ids = model.generate(
                                input_ids=input_ids,
                                max_length=84,
                                no_repeat_ngram_size=2,
                                num_beams=4
                            )[0]

    summary = tokenizer.decode(
                                output_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False
                            )

    return summary

def main():
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    article_text = st.text_area('Text to analyze')
    clean_text = WHITESPACE_HANDLER(article_text)
    sammary = samm_model(clean_text)
    st.write('Sentiment:', sammary)
    # st.title("Сгенерирую картинку с числом")
    # st.write('Для генерации конкретных картинок использовалась CGAN')
    # count = st.slider('Количество цифр:', 1, 10, 1)
    # num = st.slider('Число:', 0, 9, 1)
    # st.image(load_image(generate(count, num)), width=100*count)


if __name__ == '__main__':
    init_model()
    main()