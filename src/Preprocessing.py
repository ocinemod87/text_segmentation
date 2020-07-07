from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import re

class Preprocessing:

    def __init__(self):
        # replace urls
        self.re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                            .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                            re.MULTILINE|re.UNICODE)
        # replace ips
        self.re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

        # setup tokenizer
        self.tokenizer = WordPunctTokenizer()

        self.vocab = Counter()

    def text_to_wordlist(self, text, lower=False):
        #text = re.sub("\d+", "num", text)
        # replace URLs
        text = re.sub("http\S+", "URL", text)
        #text = re.sub(r'\*+', '*', text)
        #text = re.sub(r'\b\w{100,}\b', 'long_word', text)

        #text = re.sub("\w*cgltf\w*", "cgltf", text)

        #text = re.sub(r'\*+', '*', text)
        #text = text.replace("*", "")

        # replace IPs
        text = self.re_ip.sub(r"IPADDRESS", text)

        # replace E-mails
        #text = re.sub('([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', r'email', text)

        #text = re.sub("[0-9]+,[0-9]+", "num", text )

        # Tokenize
        text = self.tokenizer.tokenize(text)

        # optional: lower case
        if lower:
            text = [t.lower() for t in text]

        # Return a list of words
        self.vocab.update(text)
        return text

    def process_text(self, list_sentences, lower=False):
        comments = []
        for text in list_sentences:
            txt = self.text_to_wordlist(text, lower=lower)
            comments.append(txt)
        return comments, self.vocab
