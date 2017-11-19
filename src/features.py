from sklearn.base import BaseEstimator
import spacy
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
nlp = spacy.load('en')


def fetchlist(file):
    filecontent = open(file)
    lines = filecontent.readlines()
    l = []
    for line in lines:
        l.append(line.strip())
    # print(list)
    return l


def customfind(sent, phrase):
    pp = phrase.split(" ")
    if len(pp) > 1:
        if sent.find(phrase) != -1:
            return True;
    else:
        words = sent.split(" ")
        for word in words:
            if (word == phrase):
                return True
    return False


class CleanTransformer(BaseEstimator):
    def __init__(self):
        self.badlist = fetchlist('sortedBadWords_for_checking.txt')
        self.goodlist = fetchlist('positives.txt')

    #         with open('badlist.txt') as file:
    #             badwords = [line.strip() for line in file.readlines()]
    #         self.badwords = badwords

    def get_feature_names(self):
        return np.array(['bcount', 'bratio', 'gcount', 'gratio', 'bgratio', 'len_ratio',
                         'caps_ratio', 'off_score', 'lex1', 'lex2'])

    def fit(self):
        return self

    def transform(self, docs):
        bcount = []
        bratio = []
        gcount = []
        gratio = []
        bgratio = []
        len_ratio = []
        caps_ratio = []
        off_score = []
        lex1 = []
        lex2 = []

        for i, doc in enumerate(docs):
            print(i, doc)
            #             bcount_ ,bratio_ = self._bad_count(doc, self.badlist)
            #             bcount.append(bcount_)
            #             bratio.append(bratio_)

            #             gcount_ ,gratio_ = self._good_count(doc, self.goodlist)
            #             gcount.append(gcount_)
            #             gratio.append(gratio_)

            #             bgratio.append(bcount_ / float(gcount_))

            #             len_ratio.append(self._sent_len_ratio(doc, 50))

            #             caps_ratio.append(self._caps_ratio(doc))
            send_list = [0, 1.5, .6, 0, 1, .4, .6, .3, .15, .8, .45]
            off_score.append(self._offense_score(doc, send_list, self.badlist, self.goodlist))
        # lex1.append(self._lex_score1(doc, self.badlist))
        #             lex2.append(self._lex_score2(doc, self.badlist))



        #             n_words.append(len(doc.split()))
        #             n_chars.append(len(doc))
        #             caps.append()
        #             caps_ratio.append(np.sum([c.isupper() for c in doc]) / len(doc))

        return np.array([bcount, bratio, gcount, gratio,
                         bgratio, len_ratio, caps_ratio, off_score, lex1, lex2]).T

    def _bad_count(self, document, badlist):
        count = 0
        for phrase_ in badlist:
            if customfind(document, phrase_):
                # print(phrase_)
                count = count + 1
        wl = len(word_tokenize(document))
        if wl == 0:
            print(document)
        return [count, count * 10 / float(wl)]

    def _good_count(self, document, goodlist):
        count = 1
        for phrase_ in goodlist:
            if customfind(document, phrase_):
                count = count + 1
        return [count, count * 10 / float(word_tokenize(document).__len__())]

    def _sent_len_ratio(self, sentence, shortis):
        short_sent = 0
        sent_tokens = sent_tokenize(sentence)
        for sent in sent_tokens:
            if len(sent) <= shortis:
                short_sent = short_sent + 1
        return float(short_sent) / sent_tokens.__len__()

    def _caps_ratio(self, document):
        uc_token = 0
        w_tokens = word_tokenize(document)
        words_analysed = 0
        # print(w_tokens)
        for word in w_tokens:
            # print(word)
            if len(word) > 1 and (
                                                            word != "'s" and word != "'S" and word != "'ll" and word != "'LL" and word != "n't" and word != "N'T" and word != "'re" and word != "'RE" and word != "'d" and word != "'D" and word != "'ve" and word != "'VE"):
                words_analysed = words_analysed + 1
                if word.isupper():
                    uc_token = uc_token + 1
                    # print(word, words_analysed, uc_token)
        val = uc_token / float(words_analysed)
        return val

    def _offense_score(self, document, in_list, badlist, goodlist):
        s1 = in_list[0]
        s2 = in_list[1]
        s3 = in_list[2]
        s4 = in_list[3]
        s5 = in_list[4]
        s6 = in_list[5]
        s7 = in_list[6]
        s8 = in_list[7]
        s9 = in_list[8]
        s10 = in_list[9]
        s11 = in_list[10]
        # s1 -for not in sibling -0
        # s2 -for you in sibling -1.5
        # s3 -for they in sibling -0.6
        # s4 -for 'nor, neither' in descendant -0
        # s5 -for you in descendant -1
        # s6 -for they in descendant -0.4
        # s7 -for you in parent's child -.6
        # s8 - for they in parent's child - .3
        # s9 - for otherwise -.2
        # s10 -for niece you - .8
        # s11 -for niece they - .45

        wordlist = ['yourself', 'your', 'yours', 'you', 'he', 'her', 'she', 'his', 'they', 'their', 'them', 'not',
                    'never', 'nobody', 'neither', 'nor', 'it', 'its']
        document = document.lower()
        doc = nlp(document)
        sentences_in_doc = doc.sents
        # print(doc)
        Flag = False
        # displacy.serve(doc, style='dep')
        score = 0
        token_no = 0
        for token in doc:
            # check if token is a curse word
            Flag = False
            stemmedword = ps.stem(token.text)
            if token_no + 1 < len(doc) and doc[token_no + 1].text in goodlist:
                Flag = True
            else:
                break
            if Flag:
                continue
            for child in token.head.children:
                if child.text == 'not' or child.text == 'never' or child.text == 'nobody':
                    score = score + s1
                    Flag = True
                    break
            if Flag: continue
            for child in token.head.children:

                if child.text == 'you' or child.text == 'yourself' or child.text == 'your' or child.text == 'yours' or child.text == 'it' or child.text == 'its':
                    score = score + s2
                    # print("Rule--",child)
                    Flag = True
                    break
            if Flag: continue
            for child in token.head.children:
                if child.text == 'they' or child.text == 'he' or child.text == 'she' or child.text == 'her' or child.text == 'their' or child.text == 'them' or child.text == 'his':
                    score = score + s3
                    Flag = True
                    break
            if Flag: continue

            # analysing 'descendant' relation
            for node in token.children:
                if node.text == 'neither' or node.text == 'nor':  # node.text=='not' or node.text='never' or
                    score = score + s4  # first descendant relation
                    Flag = True
                    break
                else:
                    for nodechild in node.children:
                        if nodechild.text == 'neither' or nodechild.text == 'nor':
                            score = score + (s4 / 2)  # grandchild descendant so score is halved
                            Flag = True
                            break
            if Flag: continue
            for node in token.children:
                if node.text == 'you' or child.text == 'yourself' or node.text == 'your' or node.text == 'yours' or node.text == 'it' or node.text == 'its':
                    score = score + s5
                    # print("Rule--",node)
                    Flag = True
                    break;
                else:
                    for nodechild in node.children:
                        if nodechild.text == 'you' or child.text == 'yourself' or nodechild.text == 'your' or nodechild.text == 'yours' or nodechild.text == 'it' or nodechild.text == 'its':
                            score = score + (s5 / 2)
                            # print("Rule--",node)
                            Flag = True
                            break
            if Flag: continue
            for node in token.children:
                if node.text == 'they' or node.text == 'he' or node.text == 'she' or node.text == 'her' or node.text == 'their' or node.text == 'them' or node.text == 'his':
                    score = score + s6
                    Flag = True
                    break;
                else:
                    for nodechild in node.children:
                        if nodechild.text == 'they' or nodechild.text == 'he' or nodechild.text == 'she' or nodechild.text == 'her' or nodechild.text == 'their' or nodechild.text == 'them' or nodechild.text == 'his':
                            score = score + (s6 / 2)
                            Flag = True
                            break
            if Flag: continue
            # analysing niece relation
            father = token.head
            # print("pohcha",father)
            for child in father.children:
                if child != token:
                    for grandchild in child.children:
                        if grandchild.text == 'you' or child.text == 'yourself' or grandchild.text == 'your' or grandchild.text == 'yours' or grandchild.text == 'it' or grandchild.text == 'its':
                            score = score + s10
                            # print("Rules--",child)
                            Flag = True
                            break;
            if Flag: continue
            for child in father.children:
                if child != token:
                    for grandchild in child.children:
                        if grandchild.text == 'they' or grandchild.text == 'he' or grandchild.text == 'she' or grandchild.text == 'her' or grandchild.text == 'their' or grandchild.text == 'them' or grandchild.text == 'his':
                            score = score + s11
                            # print("Rules--",child)
                            Flag = True
                            break;
            if Flag: continue
            # analysing 'UNCLE' relation
            if father.dep_ != 'ROOT':
                for desc in father.head.children:
                    if desc.text == 'you' or child.text == 'yourself' or desc.text == 'your' or desc.text == 'yours' or desc.text == 'it' or desc.text == 'its':
                        score = score + s7
                        # print("Rule--",desc)
                        Flag = True
                        break;
                if Flag: continue
                for desc in father.head.children:
                    if desc.text == 'they' or desc.text == 'he' or desc.text == 'she' or desc.text == 'her' or desc.text == 'their' or desc.text == 'them' or desc.text == 'his':
                        score = score + s8
                        Flag = True
                        break;
            if Flag:
                continue
            else:
                score = score + s9
                # print("Rule--last")
            token_no = token_no + 1
        return score

    def _lex_score1(self, document, badlist):
        nlp.vocab["you"].is_stop = False
        nlp.vocab["your"].is_stop = False
        nlp.vocab["yours"].is_stop = False
        nlp.vocab["yourself"].is_stop = False
        document = document.lower()
        sentences_l = sent_tokenize(document)
        # doc=nlp(document)
        lexscore = 0
        # sent_count=0
        for sentence in sentences_l:
            doc = nlp(sentence)
            # print(doc)
            badindexes = []
            youindexes = []
            for i in range(0, doc.__len__()):
                #         token=doc[i].lemma_
                # print(doc[i],"---",doc[i].is_stop)
                if (not doc[i].is_stop):
                    if doc[i].text == 'you' or doc[i].text == 'your' or doc[i].text == "your's" or doc[i].text == 'yours' or doc[i].text == 'yourself':
                        # print(i)
                        youindexes.append(i)
                        continue
                    if badlist.__contains__(doc[i].text):
                        badindexes.append(i)
                        continue
            for badindex in badindexes:
                for youindex in youindexes:
                    lexscore = lexscore + abs(badindex - youindex)
        nlp.vocab["you"].is_stop = True
        nlp.vocab["your"].is_stop = True
        nlp.vocab["yours"].is_stop = True
        nlp.vocab["yourself"].is_stop = True
        return lexscore / float(sentences_l.__len__())

    def _lex_score2(self, document, badlist):
        nlp.vocab["you"].is_stop = False
        nlp.vocab["your"].is_stop = False
        nlp.vocab["yours"].is_stop = False
        nlp.vocab["yourself"].is_stop = False
        document = document.lower()
        #     sentence=nlp(document)
        lexscore = 0
        doc = nlp(document)
        #     sent_count=0
        #         sent_count=sent_count+1

        # print(doc)
        badindexes = []
        youindexes = []
        for i in range(0, doc.__len__()):
            #         token=doc[i].lemma_
            # print(doc[i],"---",doc[i].is_stop)
            if (not doc[i].is_stop):
                if doc[i].text == 'you' or doc[i].text == 'your' or doc[i].text == "your's" or doc[i].text == 'yours' or doc[i].text == 'yourself':
                    # print(i)
                    youindexes.append(i)
                    continue
                if doc[
                    i].text in badlist:
                    badindexes.append(i)
                    continue
        for badindex in badindexes:
            for youindex in youindexes:
                lexscore = lexscore + abs(badindex - youindex)
        nlp.vocab["you"].is_stop = True
        nlp.vocab["your"].is_stop = True
        nlp.vocab["yours"].is_stop = True
        nlp.vocab["yourself"].is_stop = True
        return lexscore / float(doc.__len__())
