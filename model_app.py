#import transformers
import nltk
from sklearn.metrics import fbeta_score
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BartTokenizer


# Load the trained t5-model and tokenizer

t5_model_path = "/Users/srinivaslakshman/Documents/1.RequiredDocuments/Course/453/Grammar_Correction_project/English_Grammar_Error/trained_t5_model" 
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)


def getCorrectAnswerUsingT5(incorrect_sentence : str, input_correct_sentence : str):
 input_text = incorrect_sentence
 input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")

 outputs = t5_model.generate(input_ids)
 t5_corrected_sentence = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

 reference = nltk.word_tokenize(input_correct_sentence)
 hypothesis = nltk.word_tokenize(t5_corrected_sentence)
 t5_bleu_score = nltk.translate.bleu_score.corpus_bleu(reference, hypothesis)
 return t5_corrected_sentence, t5_bleu_score




