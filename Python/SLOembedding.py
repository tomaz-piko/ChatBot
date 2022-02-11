#!/usr/bin/env python
# coding: utf-8

# In[121]:


import os
import sys
import re
import numpy as np
#from gensim.models import Word2Vec
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.preprocessing.text import Tokenizer

#from sklearn.model_selection import train_test_split


# In[122]:


def my_subtitles_processing(max_word_count=25):
    lines_filtered = []
    subtitles = [
        '10 Things I Hate About You.srt',
        'Back To The Future.srt',
        'Call Me By Your Name.srt',
        'Catch Me If You Can.srt',
        'Ferris Buellers Day Off.srt',
        'Fight Club.srt',
        'Gladiator.srt',
        'Inception.srt',
        'Love Simon.srt',
        'Mission Impossible Fallout.srt',
        'Mission Impossible Ghost Protocol.srt',
        'Mission Impossible Rogue Nation.srt',
        'Ready Player One.srt',
        'Shawshank Redemption.srt',
        'The Breakfast Club.srt',
        'The Hangover.srt',
        'The Matrix Reloaded.srt',
        'The Matrix Revolutions.srt',
        'The Matrix.srt',
        'Titanic.srt'
    ]
    text = ''
    for i in subtitles:
        fileR = open('Subtitles/' + i, 'r')
        lines = fileR.readlines()
        index = 1
        j = 0
        while j < len(lines) - 1:
            if lines[j].replace('\n', '') == str(index):
                j = j + 1
                index += 1
            elif lines[j] == '\n':
                j = j + 1
                continue
            else:
                #print(lines[j].replace('\n', ' '))
                if not lines[j].isupper():
                    text = text + lines[j].replace('\n', ' ').replace('<i>', '').replace('</i>', '')
            j = j + 1
        lines = nltk.sent_tokenize(text)
        for l in lines:
            line = re.sub(
                            pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]–', 
                            repl='', 
                            string=l).lower()
            line += ' ' + l[-1]
            line = line.split(' ')
            line.insert(0, '<START>')
            line.append('<END>')
            lines_filtered.append(line)
        fileR.close()
    return lines_filtered


# In[123]:


def lines_from_subtitles2(tokenize=True, max_word_count=30):
    
    """ subtitles = [
        '10 Things I Hate About You.srt',
        'Back To The Future.srt',
        'Call Me By Your Name.srt',
        'Catch Me If You Can.srt',
        'Ferris Buellers Day Off.srt',
        'Fight Club.srt',
        'Gladiator.srt',
        'Inception.srt',
        'Love Simon.srt',
        'Mission Impossible Fallout.srt',
        'Mission Impossible Ghost Protocol.srt',
        'Mission Impossible Rogue Nation.srt',
        'Ready Player One.srt',
        'Shawshank Redemption.srt',
        'The Breakfast Club.srt',
        'The Hangover.srt',
        'The Matrix Reloaded.srt',
        'The Matrix Revolutions.srt',
        'The Matrix.srt',
        'Titanic.srt'
    ]
    subtitles = ['10 Things I Hate About You.txt']
    
    def is_valid_line(line):
        if line == '\n':
            return false #prazna vrstica.
        elif '-->' in line: 
            return false #vrstica, ki vsebuje timestamp
        elif line[:-1].isnumeric():
            return False #označuje številko vrstice/prevoda
        #elif line.find('<i>') > -1 or line.find('<i>') > -1:
            #return False #monologi
        elif line.isupper():
            return False #celotna poved uppercase, ponavadi označuje kraje, prevode na tablah itd.
        else:
            return True"""
    
    lines_processed = []
    words = []
    #predlogi = ['k', 'h', 's', 'z', 'o', 'v']
    line = ''
    for s in subtitles:
        file = open('./Subtitles_processed/' + s, 'r')
        file_lines = file.readlines()
        print(file_lines[:10])
        return 0, file_lines[:10]
        break
        for i in file_lines:
            if is_valid_line(i):
                if i[-2] == '.' or i[-2] == '?' or i[-2] == '!':
                    line += i.replace('\n', '')
                    [words.append(word.lower()) for word in re.findall('\w+', line) if len(word) > 1 or word in predlogi]
                    if tokenize:
                        line_list = re.findall('\w+|[.!?]', line.lower()) #Obdrzi vejice in pike / pobere le besede.
                        if not len(line_list) > max_word_count:
                            line_list.insert(0, '<START>')
                            line_list.append('<END>')
                            lines.append(line_list)
                    else:
                        lines.append(line)
                    line = ''
                elif i[-1] == '\n':
                    line += i.replace('\n', ' ')
                
    
    #return lines[1:]
    return words, lines
    #return words, lines[1:-2:2], lines[2:-1:2]


# In[124]:


def lines_from_subtitles(tokenize=True, max_word_count=30):
    
    subtitles2 = [
        '10 Things I Hate About You.txt',
        'Back To The Future.txt',
        'Call Me By Your Name.txt',
        'Catch Me If You Can.txt',
        'Ferris Buellers Day Off.txt',
        'Fight Club.txt',
        'Gladiator.txt',
        'Inception.txt',
        'Love Simon.txt',
        'Mission Impossible Fallout.txt',
        'Mission Impossible Ghost Protocol.txt',
        'Mission Impossible Rogue Nation.txt',
        'Ready Player One.txt',
        'Shawshank Redemption.txt',
        'The Breakfast Club.txt',
        'The Hangover.txt',
        'The Matrix Reloaded.txt',
        'The Matrix Revolutions.txt',
        'The Matrix.txt',
        'Titanic.txt'
    ]
    
    subtitles = [
        #'Bad Boys For Life.txt',
        #'Crawl.txt',
        #'Dirty Grandpa.txt',
        #'Ford V Ferrari.txt',
        #'Little Women.txt',
        #'Passengers.txt',
        #'The Beach Bum.txt',
        #'The invisible man.txt',
        #'The_Gentlemen.txt',
        #'The_Irishman.txt',
        'PotC Curse Of The Black Pearl.txt',
        'PotC Dead Mans Chest.txt',
        'PotC At Worlds End.txt'
    ]
    
    def is_valid_line(line):
        if line == '\n':
            return False #prazna vrstica.
        elif '-->' in line: 
            return False #vrstica, ki vsebuje timestamp
        elif line[:-1].isnumeric():
            return False #označuje številko vrstice/prevoda
        #elif line.find('<i>') > -1 or line.find('<i>') > -1:
            #return False #monologi
        elif line.isupper():
            return False #celotna poved uppercase, ponavadi označuje kraje, prevode na tablah itd.
        else:
            return True
    
    lines_processed = []
    words = []
    text = ''
    fileW = open('./Subtitles_processed/1.txt', 'w')
    for s in subtitles:
        fileR = open('./Subtitles_processed/subs2/' + s, 'r')       
        file_lines = fileR.readlines()
        for l in file_lines:
            if is_valid_line(l): 
                l = l.replace('<i>', '')
                l = l.replace('</i>', '')
                l = l.replace('\n', ' ')
                text += l
                fileW.write(l + '\n')
        fileR.close()
    fileW.close()

    sentences = nltk.tokenize.sent_tokenize(text)
    
    fileW = open('./Subtitles_processed/2.txt', 'w')
    for s in sentences:
        if ' -' in s:
            s = s.replace('...', '.')
            lines = s.split(' -')
            for l in lines:
                if len(l) > 0:  
                    lines_processed.append(l)
                    fileW.write(l + '\n')
        else:
            s = s.replace('...', '')
            lines_processed.append(s)
            fileW.write(s + '\n')
    fileW.close()
    
    lines_final = []
    fileW = open('./Subtitles_processed/3.txt', 'w')
    for l in lines_processed:
        if ':' in l:
            idx = l.find(':')
            if not l[idx-1].isdigit() and not l[idx+1].isdigit():
                l = l.replace(':', '')
        l = re.sub('[-,\"()_\']', '', l)
        l = l[:-1] + ' ' + l[-1] #Zadnje ločilo zamaknemo za eno v desno, da lahko splitamo po presledkih
        l = l.lower()
        lines_final.append(l)
        fileW.write(l + '\n')
               
    return text, lines_final


# In[125]:


fileSLO = open('./DataSLO/Final.txt', 'r', encoding='utf-8')
fileENG = open('./DataENG/lines_final.txt', 'r')
linesSLO = fileSLO.readlines()
linesSLO = [l.split() for l in linesSLO]
linesENG = fileENG.readlines()[50000:100000:1]
linesENG = [l.split() for l in linesENG]


# In[126]:


max_length_slo = max([len(l) for l in linesSLO])
print("Line count: " + str(len(linesSLO)))
print("Max length: " + str(max_length_slo))
count_matrix_slo = np.zeros((max_length_slo + 1), dtype='int32')
for l in linesSLO:
    count_matrix_slo[len(l)] += 1
avg = 0
for i in range(1, len(count_matrix_slo), 1):
    avg += i * count_matrix_slo[i]
avg = avg // len(linesSLO)
print("Average length: " + str(avg))

sum_matrix_slo = [
                sum(count_matrix_slo[:5]),
                sum(count_matrix_slo[5:10]),
                sum(count_matrix_slo[10:15]),
                sum(count_matrix_slo[15:20]),
                sum(count_matrix_slo[20:25]),
                sum(count_matrix_slo[25:30]),
                sum(count_matrix_slo[30:])
             ]
print(sum_matrix_slo)
avg_matrix_slo = []
for s in sum_matrix_slo:
    avg_matrix_slo.append(round((s / len(linesSLO) * 100), 2))
print(avg_matrix_slo)

N = len(avg_matrix_slo)
ind = np.arange(N)
plt.figure(figsize=(10, 5))
plt.bar(ind, avg_matrix_slo)
plt.title('Odstotek povedi v podatkovni zbirki glede na dolžino povedi.', fontsize=16)
plt.ylabel('Odstotek (%)', fontsize=12)
plt.xlabel('Št. besed v povedi', fontsize=12)
plt.xticks(np.arange(0, N, 1), ('1 do 5', '5 do 10', '10 do 15', '15 do 20', '20 do 25', '25 do 30', '30 in več'), fontsize=11)
plt.show()


# In[127]:


max_length_eng = max([len(l) for l in linesENG])
print("Line count: " + str(len(linesENG)))
print("Max length: " + str(max_length_eng))
count_matrix_eng = np.zeros((max_length_eng + 1), dtype='int32')
for l in linesENG:
    count_matrix_eng[len(l)] += 1
avg = 0
for i in range(1, len(count_matrix_eng), 1):
    avg += i * count_matrix_eng[i]
avg = avg // len(linesENG)
print("Average length: " + str(avg))

sum_matrix_eng = [
                sum(count_matrix_eng[:5]),
                sum(count_matrix_eng[5:10]),
                sum(count_matrix_eng[10:15]),
                sum(count_matrix_eng[15:20]),
                sum(count_matrix_eng[20:25]),
                sum(count_matrix_eng[25:30]),
                sum(count_matrix_eng[30:])
             ]
print(sum_matrix_eng)
avg_matrix_eng = []
for s in sum_matrix_eng:
    avg_matrix_eng.append(round((s / len(linesSLO) * 100), 2))
print(avg_matrix_eng)

N = len(avg_matrix_eng)
ind = np.arange(N)
plt.figure(figsize=(10, 5))
plt.bar(ind, avg_matrix_eng)
plt.title('Odstotek povedi v podatkovni zbirki glede na dolžino povedi.', fontsize=16)
plt.ylabel('Odstotek (%)', fontsize=12)
plt.xlabel('Št. besed v povedi', fontsize=12)
plt.xticks(np.arange(0, N, 1), ('1 do 5', '5 do 10', '10 do 15', '15 do 20', '20 do 25', '25 do 30', '30 in več'), fontsize=11)
plt.show()


# In[128]:


from gensim.models import Word2Vec
fileR = open('./DataSLO/Final.txt', 'r', encoding='utf-8')
lines = fileR.readlines()
model = Word2Vec(lines,
                min_count=3,
                size=100,
                workers=8,
                window=3,
                iter=15)
model.wv.save_word2vec_format('./DataSLO/embedding_SLO_100_3.txt', binary=False)
fileR.close()


# In[129]:


model.wv.save_word2vec_format('./DataSLO/embedd_tokenless_punctless_200_2', binary=False)


# # EN MODEL DVA GRAFA

# In[130]:


path = 'SLO50_BS32_LD128_ES100_LC25.0k_VS15000'
history = np.load(path + '/history.npy',allow_pickle='TRUE').item()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(history['accuracy'])
ax1.plot(history['val_accuracy'])
ax1.set_title('Natančnost modela', fontsize=16)
ax1.set_xlabel('epoha', fontsize=12)
ax1.set_ylabel('natančnost', fontsize=12)
ax1.legend(['train_accuracy', 'val_accuracy'], loc='upper left', fontsize=12)
ax2.plot(history['loss'])
ax2.plot(history['val_loss'])
ax2.set_title('Izguba modela', fontsize=16)
ax2.set_xlabel('epoha', fontsize=12)
ax2.set_ylabel('izguba', fontsize=12)
ax2.legend(['train_loss', 'val_loss'], loc='upper left', fontsize=12)
plt.show()


# # DVA MODELA DVA GRAFA

# In[131]:


paths = ['SLO50_BS32_LD128_ES100_LC25.0k_VS15000', './ENG50_BS32_LD128_ES100_LC25.0k_VS15000']
histories = []
for path in paths:
    history = np.load(path + '/history.npy',allow_pickle='TRUE').item()
    histories.append(history)
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for history in histories:    
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title('Natančnosti modelov', fontsize=16)
    #ax1.set(xlabel='epoch', ylabel = 'accuracy')
    ax1.set_xlabel('epoha', fontsize=12)
    ax1.set_ylabel('natančnost', fontsize=12)
    ax1.legend(
        ['SLO train_accuracy',
         'SLO val_accuracy',
         'ENG train_accuracy',
         'ENG val_accuracy'
        ], loc='upper left')
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Izgube modelov', fontsize=16)
    #ax2.set(xlabel='epoch', ylabel='loss')
    ax2.set_xlabel('epoha', fontsize=12)
    ax2.set_ylabel('izguba', fontsize=12)
    ax2.legend(
        ['SLO train_loss',
         'SLO val_loss',
         'ENG train_loss',
         'ENG val_loss'
        ], loc='upper left')

plt.show()


# # VEČ MODELOV ŠTIRI GRAFI

# In[132]:


paths = ['./SLO50_BS16_LD128_ES100_LC25.0k_VS15000',
         './SLO50_BS32_LD128_ES100_LC25.0k_VS15000',
         './SLO_BS64_LD128_ES100_LC25.0k_VS15000',
         './SLO_BS128_LD128_ES100_LC25.0k_VS15000'
        ]
histories = []
for path in paths:
    history = np.load(path + '/history.npy',allow_pickle='TRUE').item()
    histories.append(history)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 11))
plt.subplots_adjust(wspace=None, hspace=0.25)
for history in histories:
    ax1.plot(history['accuracy'])
    ax2.plot(history['val_accuracy'])
    ax1.set_title('Natančnost modelov na učni množici', fontsize=16)
    ax2.set_title('Natančnost modelov na testni množici', fontsize=16)
    #ax1.set(xlabel='epoch', ylabel = 'train_accuracy')
    ax1.set_xlabel('epoha', fontsize=13)
    ax1.set_ylabel('natančnost', fontsize=13)
    #ax2.set(xlabel='epoch', ylabel = 'val_accuracy')
    ax2.set_xlabel('epoha', fontsize=13)
    ax2.set_ylabel('natačnost', fontsize=13)
    
for history in histories:
    ax3.plot(history['loss'])
    ax4.plot(history['val_loss'])
    ax3.set_title('Izguba modelov na učni množici', fontsize=16)
    ax4.set_title('Izguba modelov na testni množici', fontsize=16)
    #ax3.set(xlabel='epoch', ylabel='train_loss')
    ax3.set_xlabel('epoha', fontsize=13)
    ax3.set_ylabel('izguba', fontsize=13)
    #ax4.set(xlabel='epoch', ylabel='val_loss')
    ax4.set_xlabel('epoha', fontsize=13)
    ax4.set_ylabel('izguba', fontsize=13)
    
param = 'velikost paketa'
params = [16, 32, 64, 128]
legend_params = []
for p in params:
    #legend_params.append(param + ' = ' + str(p))
    legend_params.append(str(p))
    
legend = plt.legend(legend_params,
          bbox_to_anchor=(1.45, 2), fontsize=12, title='Velikost paketa:')
plt.setp(legend.get_title(),fontsize='14')
plt.show()


# In[134]:


paths = ['./ENG50_BS32_LD128_ES100_LC25.0k_VS15000',
         './ENG50_BS32_LD128_ES100_LC50.0k_VS15000',
         './ENG50_BS32_LD128_ES100_LC75.0k_VS15000'
        ]
histories = []
for path in paths:
    history = np.load(path + '/history.npy',allow_pickle='TRUE').item()
    histories.append(history)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 11))
plt.subplots_adjust(wspace=None, hspace=0.25)
for history in histories:
    ax1.plot(history['accuracy'])
    ax2.plot(history['val_accuracy'])
    ax1.set_title('Natančnost modelov na učni množici', fontsize=16)
    ax2.set_title('Natančnost modelov na testni množici', fontsize=16)
    #ax1.set(xlabel='epoch', ylabel = 'train_accuracy')
    ax1.set_xlabel('epoha', fontsize=13)
    ax1.set_ylabel('natančnost', fontsize=13)
    #ax2.set(xlabel='epoch', ylabel = 'val_accuracy')
    ax2.set_xlabel('epoha', fontsize=13)
    ax2.set_ylabel('natačnost', fontsize=13)
    
for history in histories:
    ax3.plot(history['loss'])
    ax4.plot(history['val_loss'])
    ax3.set_title('Izguba modelov na učni množici', fontsize=16)
    ax4.set_title('Izguba modelov na testni množici', fontsize=16)
    #ax3.set(xlabel='epoch', ylabel='train_loss')
    ax3.set_xlabel('epoha', fontsize=13)
    ax3.set_ylabel('izguba', fontsize=13)
    #ax4.set(xlabel='epoch', ylabel='val_loss')
    ax4.set_xlabel('epoha', fontsize=13)
    ax4.set_ylabel('izguba', fontsize=13)
    
param = 'velikost paketa'
params = [50000, 100000, 150000]
legend_params = []
for p in params:
    #legend_params.append(param + ' = ' + str(p))
    legend_params.append(str(p))
    
legend = plt.legend(legend_params,
          bbox_to_anchor=(1.35, 2), fontsize=12, title='Št. povedi:')
plt.setp(legend.get_title(),fontsize='14')
plt.show()


# In[ ]:




