import re


def preprocessing(sentence, pathofbadwords):
    cleanr = re.compile('<.*?>')

    regex = "[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    regex1 = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][" \
             "a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2," \
             "}|www\.[a-zA-Z0-9]\.[^\s]{2,}) "
    regex2 = "(www | http: | https:)+[ ^\s]+[\w]"

    badfile = open(pathofbadwords, 'r')
    bad_word_dict = dict()
    for line in badfile:
        bw = line.split(',')
        if len(bw) == 2:
            # print(bw[0])
            # print(bw[1])
            bad_word_dict[bw[0]] = bw[1].strip()
            # print(bw[0])
            # print(bw[1].strip())

    s0 = sentence

    s0 = s0.lower()
    s0 = s0.replace("\\\\n", " ")
    s0 = s0.replace("\\n", " ")
    s0 = s0.replace("\\t", " ")
    s0 = s0.replace("\\\\xc2", " ")
    s0 = s0.replace("\\\\xa0", " ")
    s0 = s0.replace("\\\\xa0", " ")
    s0 = s0.replace("\\[\\w]", ' ')
    s0 = re.sub(r"\\[a-zA-Z0-9.]*", "", s0)

    s0 = re.sub("([a-zA-Z0-9.?!#*])\\1\\1+", "\\1", s0)  # brooooook->brook

    s0 = re.sub(regex, "", s0)  # url
    s0 = re.sub(regex2, "", s0)  # http url
    s0 = re.sub(regex1, "", s0)
    s0 = re.sub(cleanr, '', s0)  # html tags

    # string = ":-/)"
    # ##  REMOVING SMILEYS
    # s0=re.sub(string,"  smiley",s0);
    # # s0=re.sub("\[\]+","",s0)            #remove \
    s0 = s0.strip()
    s0 = s0.replace(" wont ", " will not ")
    s0 = s0.replace(" won't ", " will not ")
    s0 = s0.replace(" don't ", " do not ")
    s0 = s0.replace(" didn't ", " did not ")
    s0 = s0.replace("Didn't ", "Did not ")
    s0 = s0.replace(" i'll", " I will")
    s0 = s0.replace(" I'll", " I will")
    s0 = s0.replace("I'll", "I will")
    s0 = s0.replace(" can't", " cannot")
    s0 = s0.replace(" shouldn't", " should not")
    s0 = s0.replace(" im ", " i am ")
    s0 = s0.replace("ain't", "is not")
    s0 = s0.replace("'ll", " will")
    s0 = s0.replace("'t[. ]", " not")
    # s0=s0.replace(" u ", " you ")
    s0 = s0.replace(" r ", " are ")
    s0 = s0.replace(" m ", " am ")
    s0 = s0.replace(" u'r ", " you are ")
    s0 = s0.replace(" you'r ", "you are ")
    s0 = s0.replace("'ve", " have")
    s0 = s0.replace("'s", " is")
    s0 = s0.replace("'re", " are")
    s0 = s0.replace("'d", " would")
    s0 = re.sub("([a-zA-Z0-9.]+)\\1\\1+", " ", s0)  # lolololol->lol
    s0 = re.sub("[&*?!#^%`~$@]{4}", "-TOKEN-", s0)  # &*$!^@->>>>token
    s0 = s0.strip();

    # print("before       " + s0)

    for key, value in bad_word_dict.items():
        sk = s0.replace(" " + key, " " + value + " ")
        if sk != s0:
            s0 = sk

    s0 = re.sub("(@|#)[\w.]*", "-PRON-", s0)  # @username with YOU

    return s0
