import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 1
BERT_PATH = "/media/cruigo/stuff2/projects/sentiment_bert/data/bert-base-uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../data/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)