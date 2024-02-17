#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:53:10 2023

@author: allen82218
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:44:03 2022

@author: anpo
"""

import numpy as np
import pandas as pd
import re 
#from fuzzysearch import find_near_matches
import os
# import schedule
#import cc_config
#import logging# functions
import sys

def keyword_match2(note_text, allkeys, use_fuzz=False):
    # id_part = note_text[['CSN', 'CHTNO']]   
    txt_part = note_text.drop(columns = ['CSN', 'CHTNO'])
    # 清理text, txt全部放在一起
    note_text_merge = []
    for i in range(len(txt_part)):
        vv = ''
        v = txt_part.iloc[i,:].dropna()
        for i in range(len(v)):
            vv = vv+v[i]
        note_text_merge.append(vv)
    # 針對一篇, 每個關鍵字比對
    #per_txt = note_text_merge[0]
   
    # 清理key
    allkeys = allkeys.apply(lambda x : x.strip())
    allkeys = list(allkeys)
    # 清理錯字
    pattern_c = r'\b(wound dehiscence|wound dehisence|wound dehisence|wound dihescence)\b'
    
    #print(allkeys)
    pattern = r'\b(?:' + '|'.join(re.escape(k) for k in allkeys) + r')\b'
    Nans = []
    #matched_keys = []
    for per_text in note_text_merge:
        per_text = re.sub(pattern_c,'wound dehiscense', per_text)
        per_text = per_text.lower()
        matches = re.findall(pattern, per_text)
        if len(matches)>0: #只要有其中一個關鍵字,report cc
           ans = 'yes'
        else:
           ans = 'no'
        Nans.append(ans)
        #matched_keys.append(matches)
    return Nans, matches


#     return ans, matched_keys_per_txt
def infer2(note_text, allkeys, ks,Tb):
    print('infer: {}'.format('Infection'))
    Nans, matched_keys = keyword_match2(note_text, allkeys, use_fuzz=False)
    
   #b = pd.read_csv('C:\dotnet\infection_table.csv')
   #Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'] = Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'].apply(lambda x : x.strip())

    detected = 0 # default

    idenfied = []
    for item in matched_keys:
        if item not in idenfied:
           idenfied.append(item) 
    
    matched = []    
    for j in idenfied:
        a = Tb['prc'][Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'] == j]
        #a = Tb['prc'][Tb['0'] == j]
        if a.values[0] > ks[0]: 
           detected = 1 

    for j in idenfied:
        a = Tb['sen'][Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'] == j]
        #a = Tb['sen'][Tb['0'] == j]
        if (a.values[0] > ks[1]) & (len(idenfied)>1):
        #if (a.values[0] > ks[1]):    
           detected = 1 
                   
    if len(idenfied)>ks[2]:
       detected = 1  
       
    return detected, matched_keys


def main(input_file,allkeys,Tb):
    
    #
    #print('檔案所在的資料夾:{}'.format(os.path.dirname(input_file)))
    
    #dirname = os.path.dirname(input_file)
    # print(input_file)
    # dirname = os.path.abspath(os.path.dirname(__file__))
    #print(dirname)
    
    # initiate log file
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)
   
#     logging.basicConfig(level=logging.DEBUG, filename=os.path.join(dirname, 'cc_keyfn_log.log'), filemode='w',
# 	format='[%(asctime)s %(levelname)-8s] %(message)s',
# 	datefmt='%Y%m%d %H:%M:%S',
# 	)

#     logging.info('共病偵測log開始')
       
    # read
    with open(input_file,encoding="utf-8") as f:
         lines = f.readlines()

    # cols = ['CSN','CHTNO','type of note', 'txt']
    a_dict = {}
    a_dict['CSN'] = lines[0]
    a_dict['CHTNO'] = lines[1]
    a_dict['txt'] = ''.join(lines[2:])  
  
    out_file = lines[1].replace('\n', '')+'outputaiinf'
    
   # logging.info(out_file)
    
    note_text0 = pd.DataFrame.from_dict([a_dict])
        
    #load key
    #assert os.path.exists('0-合併症書寫內容調查-彙總版20220307.xlsx'), 'error!, cc keys must exist'
    # div_ = '骨科'
    # div_sheet_name = '骨科CC調查-Infection基隆'
    # col_name = '基隆骨科初版/調查其他院區該項衍生書寫內容'
    # out = pd.read_excel('C:\dotnet\cc_key.xlsx',sheet_name = div_sheet_name)
   
    # allkeys = out[col_name]
    # allkeys = allkeys.dropna()
    
    
    
    cc_ans, matched = infer2(note_text0, allkeys, [0.9, 0.9, 1],Tb)
    #cc_ans, matched_keys_per_txt = infer2(note_text0) 
    
    file_cont = ''
    if cc_ans == 1: 
       #file_cont += '書寫出現關鍵診斷:{}'.format(matched)
       #file_cont += '\n'
       #file_cont += '該病人可能有{}等手術合併症, 提醒您事實介入相關處置'.format('infection')      
       file_cont += 'infection'
       key_str = ''
       for i in matched:
           key_str += i
           key_str += ','
  
       file_cont += '\n'
       file_cont += key_str[:-1]

    else:
       file_cont += 'N'
        
       #logging.info(file_cont)           
       #print(file_cont)
       #print()
        # save the results
      
        # print('hello')
    with open(os.path.join('C:\TEMP',out_file+'.txt'),'w') as f:
         f.write(file_cont)
             

div_ = '骨科'
div_sheet_name = '骨科CC調查-Infection基隆'
col_name = '基隆骨科初版/調查其他院區該項衍生書寫內容'
#out = pd.read_excel('cc_key.xlsx',sheet_name = div_sheet_name)
out = pd.read_excel('C:\dotnet\cc_key.xlsx',sheet_name = div_sheet_name)

allkeys = out[col_name]
allkeys = allkeys.dropna()
 
Tb = pd.read_csv('C:\dotnet\infection_table.csv')
#Tb = pd.read_csv('infection_table.csv')
Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'] = Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'].apply(lambda x : x.strip())
    

if __name__ == '__main__':
   
   # verify the input 
   print('輸入檔案:{}'.format(sys.argv[1]))
   
  #  div_ = '骨科'
  #  div_sheet_name = '骨科CC調查-Infection基隆'
  #  col_name = '基隆骨科初版/調查其他院區該項衍生書寫內容'
  # # out = pd.read_excel('cc_key.xlsx',sheet_name = div_sheet_name)
  #  out = pd.read_excel('C:\dotnet\cc_key.xlsx',sheet_name = div_sheet_name)
  
  #  allkeys = out[col_name]
  #  allkeys = allkeys.dropna()
   
  #  Tb = pd.read_csv('C:\dotnet\infection_table.csv')
  #  #Tb = pd.read_csv('infection_table.csv')
  #  Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'] = Tb['基隆骨科初版/調查其他院區該項衍生書寫內容'].apply(lambda x : x.strip())

   
   
   main(sys.argv[1], allkeys, Tb)
   # # 執行的方式 input哪來, output那？資料清理
   #  if not os.listdir(cc_config.input_folder):  
   #     print('輸入資料夾是空的')       
   #     logging.warning('輸入資料夾是空的')
   #  else:  
       
   #     meta_data = ['12340593',33456987,'手術紀錄']
        
       # main(cc_config, meta_data)
       
     


 