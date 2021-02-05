# export LC_CTYPE=“C.UTF-8”
import sys                                                                                                                                                                                              

args = sys.argv
lexicon_path = args[1]
lexicon_outpath = args[2]

with open(lexicon_path, 'r', encoding='utf-8') as f:
    lexicon = f.readlines()

lexicon_nomorph = list()
for i in lexicon:
    i = i.strip()
    lex = i.split(' ')
    if len(lex) < 2:
        print(lex[0].encode('utf_8'), ' phone length 0. not use')
        continue
    word = lex[0]
    phone = lex[1:]
    if len(phone) > 0:
        add_line = word + ' ' + ' '.join(phone)
        lexicon_nomorph.append(add_line + "\n")
    else:
        print(word, ' phone length 0. not use')

lexicon_nomorph = list(set(lexicon_nomorph))
lexicon_nomorph = sorted(lexicon_nomorph)

with open(lexicon_outpath, 'w', encoding='utf-8') as f:
    f.writelines(lexicon_nomorph)