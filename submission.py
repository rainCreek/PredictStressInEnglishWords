
## import modules here 
import helper
from sklearn import tree, model_selection
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
from sklearn.metrics import f1_score
import nltk

################# process #################

vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
consonant = ["P","B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","R","S","SH","T","TH","V","W","Y","Z","ZH"]
attributes = ["vowel_num","two_vowel","all_vowel","last_word","suffix2","strong_pho","com1","com2","com3","com4","stress_pos"]
features = ["vowel_num","two_vowel","all_vowel","last_word","suffix2","strong_pho","com1","com2","com3","com4"]

vowel1 = {'ABLE':'AH', 'ER':'AH', 'FUL':'AH', 'ING':'IH', 'ISH':'IH', 'LESS':'IH', 
          'LY':'IH', 'MENT':'AH', 'NESS':'IH', 'OR':'AH', 'TY':'IH'}
vowel2 = {'ARY':'AH', 'ISM':'AH', 'IZE':'AY', 'ORY':'AH'}
vowel3 = {'ANCE':'AH', 'ANT':'AH', 'ENCE':'AH', 'ENT':'AH', 'EOUS':'AH', 'GRAPHY':'AH', 
        'IAL':'AH', 'IAN':'AH', 'IAR':'AH', 'IC':'IH', 'ICAL':'IH', 'ION':'AH', 'IOUS':'AH', 'LOGIST':'AH', 
        'LOGY':'AH', 'OUS':'AH', 'SIVE':'IH', 'TIVE':'IH', 'UAL':'AH', 'UOUS':'AH'}
vowel4 = {'AIN':'EH', 'EE':'IY', 'EER':'IH', 'ESE':'IH'}
strong_pho = {'A':'EY', 'E':'IY', 'I':'AY', 'O':'OW', 'U':'UW'}

# preffix 
prefixVowel = {'A':'AH', 'AB':'AH', 'AC':'AH', 'AD':'AH', 'AL':'AH', 'BE':'IH', 'CON':'AH', 'DE':'IH', 
'DIS':'IH', 'EM':'IH', 'EN':'IH', 'IN':'IH', 'MIS':'IH',  'RE':'IH', 'UN':'AH'}
# return the pos of the vowel of the suffix
def count_1(word, word_vowel):
    vowel_list_re = word_vowel.copy()
    vowel_list_re.reverse()
    for element in vowel1:
        if (word.find(element) != -1 and (vowel1[element] in word_vowel)):
            return (len(word_vowel) - vowel_list_re.index(vowel1[element]))
    return 0

def count_2(word, word_vowel):
    vowel_list_re = word_vowel.copy()
    vowel_list_re.reverse()
    for element in vowel2:
        if (word.find(element) != -1 and (vowel2[element] in word_vowel)):
            return (len(word_vowel) - vowel_list_re.index(vowel2[element]))
    return 0

def count_3(word, word_vowel):
    vowel_list_re = word_vowel.copy()
    vowel_list_re.reverse()
    for element in vowel3:
        if (word.find(element) != -1 and (vowel3[element] in word_vowel)):
            return (len(word_vowel) - vowel_list_re.index(vowel3[element]))
    return 0

def count_4(word, word_vowel):
    vowel_list_re = word_vowel.copy()
    vowel_list_re.reverse()
    for element in vowel4:
        if (word.find(element) != -1 and (vowel4[element] in word_vowel)):
            return (len(word_vowel) - vowel_list_re.index(vowel4[element]))
    return 0

def count_5(word, word_vowel):
    vowel_list_re = word_vowel.copy()
    vowel_list_re.reverse()
    for element in prefixVowel:
        if (word.find(element) != -1 and (prefixVowel[element] in word_vowel)):
            return (len(word_vowel) - vowel_list_re.index(prefixVowel[element]))
    return 0

def find_strong_pho(word, word_vowel):
    for element in strong_pho:
        if (word[-1] == element and word_vowel[-1] == strong_pho[element]):
            return len(word_vowel)
    return 0


def vowel_num(pronounce):
    vowel_num = 0
    word_vowel = []
    word_vowel_without_digit = []
    for phoneme in pronounce:
        if phoneme[:2] in vowel:
            vowel_num += 1
            word_vowel.append(phoneme)
            word_vowel_without_digit.append(phoneme[:2])
            
    return vowel_num,word_vowel,word_vowel_without_digit


def transfer_all_vowels_to_int(word_vowel):
    result = 0
    for vo in word_vowel:
        result = result * 0.01 + vowel.index( vo )+1
    result = result * pow(10, len(word_vowel))
    return result


def transfer_last2_vowels_to_int(word_vowel):
    if len(word_vowel) < 2:
        return 0
    else :
        result =  vowel.index(word_vowel[-2]) *100 + vowel.index(word_vowel[-1])
        return result


def transfer_last3_letters_to_int(word):
    letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    if word[-1] == "S" or word[-1] == "D":
        word = word[:-1]

    result = 0
    if len(word) >= 3 :
       result = ((letters.index(word[-3])+1) * 100 + (letters.index(word[-2])+1)) * 100 + letters.index(word[-1])+1
    return result

def combine_c_v_c(proList):

    combination_list = []
    for i in range(len(proList)) :
        temp = 0 
        if proList[i] in vowel:
            if i-1 >= 0 and proList[i-1] in consonant:
                temp += consonant.index(proList[i-1]) * 10000
            temp += vowel.index(proList[i]) * 100
            if i+1 < len(proList) and proList[i+1] in consonant:
                temp += consonant.index(proList[i+1]) * 1
            combination_list.append(temp)

    while len(combination_list) < 4:
        combination_list.append(0)
    #print(combination_list)

    return combination_list



################# training #################

def train(data, classifier_file):# do not change the heading of the function
    instances = []

    for line in data:
        word_vowel = []

        word = line[:line.index(":")]
        pronounce = line[line.index(":")+1:] 
        pronounceWithoutDigit = ''.join([i for i in pronounce if not i.isdigit()])
        pronounce = pronounce.split()
        pronounceWithoutDigit = pronounceWithoutDigit.split()

        #print(line, pronounce)
        # vowel_num feature
        vowel_number,word_vowel,word_vowel_without_digit = vowel_num(pronounce)

        # get features
        t_3_letter_int = transfer_last3_letters_to_int(word)
        t_all_vo_int = transfer_all_vowels_to_int(word_vowel_without_digit)
        t_2_vo_int = transfer_last2_vowels_to_int(word_vowel_without_digit)


       # c1 = count_1(word, pronounceWithoutDigit)
        c2 = count_2(word, word_vowel_without_digit)
        #c3 = count_3(word, pronounceWithoutDigit)
        #c4 = count_4(word, pronounceWithoutDigit)
        #c5 = count_5(word, pronounceWithoutDigit)
        strong_pho = find_strong_pho(word, word_vowel_without_digit)
        com_cvc = combine_c_v_c(pronounceWithoutDigit)
      
        ##get the stress position
        for vo in word_vowel:
            if "1" in vo:               
                stress = word_vowel.index(vo)+1
        
        instances.append([vowel_number,t_2_vo_int,t_all_vo_int,t_3_letter_int,c2,strong_pho,com_cvc[0],com_cvc[1],com_cvc[2],com_cvc[3],stress])
    
    df = pd.DataFrame(data=instances,columns = attributes)
    x = df[features]
    y = df.stress_pos 
    #print(x)

    clf = RandomForestClassifier( n_estimators = 40, min_weight_fraction_leaf = 0.00001,max_depth = 20)
    model=clf.fit(x, y)
    print(clf.feature_importances_)

    output = open(classifier_file,'wb')
    pickle.dump(clf,output)
    output.close()


################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    clf = pickle.load(open(classifier_file,"rb"))
    instances = []

    for line in data:
        word_vowel = []

        word = line[:line.index(":")]
        pronounce = line[line.index(":")+1:] 
        pronounceWithoutDigit = ''.join([i for i in pronounce if not i.isdigit()])
        pronounce = pronounce.split()
        pronounceWithoutDigit = pronounceWithoutDigit.split()

        #print(line, pronounce)
        # vowel_num feature
        vowel_number,word_vowel,word_vowel_without_digit = vowel_num(pronounce)

        # get features
        t_3_letter_int = transfer_last3_letters_to_int(word)
        t_all_vo_int = transfer_all_vowels_to_int(word_vowel_without_digit)
        t_2_vo_int = transfer_last2_vowels_to_int(word_vowel_without_digit)


       # c1 = count_1(word, pronounceWithoutDigit)
        c2 = count_2(word, word_vowel_without_digit)
        #c3 = count_3(word, pronounceWithoutDigit)
        #c4 = count_4(word, pronounceWithoutDigit)
        #c5 = count_5(word, pronounceWithoutDigit)
        strong_pho = find_strong_pho(word, word_vowel_without_digit)
        com_cvc = combine_c_v_c(pronounceWithoutDigit)
        
        instances.append([vowel_number,t_2_vo_int,t_all_vo_int,t_3_letter_int,c2,strong_pho,com_cvc[0],com_cvc[1],com_cvc[2],com_cvc[3]])
    

    df = pd.DataFrame(data=instances,columns = features)
    x = df[features]
    
    prediction = list(clf.predict(x))
    return prediction
  
