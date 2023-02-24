#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:44:03 2022

@author: anpo
"""

import numpy as np
import pandas as pd
import re 
from fuzzysearch import find_near_matches
import os
# import schedule
import logging# functions
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
        
        vv = vv.replace('.','')
        vv = vv.replace(',','')
        # 
        vv = vv.replace('\r','')
        vv = vv.replace('\n','')
        vv = re.sub(' +','',vv)  
        note_text_merge.append(vv)
    
    # id_part['txt_flat'] = note_text_merge
    # id_part = id_part.reset_index()
         
    # 針對一篇, 每個關鍵字比對
    matched_keys_per_txt = []
    # Nans = []
    # yes_id = []
    # for idc, rows in id_part.iterrows():
    #   per_txt = rows['txt_flat']
    per_txt = note_text_merge[0]
    # print(per_txt)
    matched_keys = []
    cnt2 = 0
    # 清理key
    for index, value in allkeys.items():
        value = value.replace('.','')
        value = value.replace(',','')
        # remove extra space 
        value = re.sub(' +','',value)
        # re search
        if (use_fuzz) & (value.lower()!='kink') :      # fuzzy search
            b = find_near_matches(value.lower(),per_txt.lower(), max_l_dist=1)               
            if b!=[]:
              matched_keys.append(value)
              cnt2+=1
              # print('hello')
        else: 
            b = re.search(value.lower(),per_txt.lower())        
            if b!=None: 
              matched_keys.append(value)
              cnt2+=1
    
    matched_keys_per_txt.append(matched_keys)           
    
    if cnt2>0: #只要有其中一個關鍵字,report cc
       ans = 'yes'
       # yes_id.append(idc)
    else:
       ans = 'no'
    # Nans.append(ans)
    # yes_ids = id_part.loc[yes_id,['CSN', 'CHTNO']] # 這次有抓到cc的iD
        
    return ans, matched_keys_per_txt


def cc_detect(note_text, series_keys):

    assert isinstance(note_text, pd.DataFrame) == True
    # assert 'CSN' in note_text
    # assert 'CHTNO' in note_text
    assert len(note_text.columns) > 2  # 除了identifier 還有其他資料
    
     # cc key detect...
    # ans, yes_ids, matched_keys_per_txt = keyword_match2(note_text, series_keys, use_fuzz=False)
    ans, matched_keys_per_txt = keyword_match2(note_text, series_keys, use_fuzz=True)
    
    return ans, matched_keys_per_txt


def main(exec_file, input_file):
    
    # parsing the paths
    dirname = os.path.dirname(exec_file)
    print('執行檔案所在的資料夾:{}'.format(dirname))
    
    assert len(dirname)>0, '無檔案路徑'
    
    dirname_file = os.path.dirname(input_file)
    print('輸入檔案所在的資料夾:{}'.format(dirname_file))
    
    assert len(dirname)>0, '無檔案路徑'
    
    # initiate log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
   
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join('cc_keyfn_log.log'), filemode='w',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)

    logging.info('共病偵測log開始')
       
    # read
    with open(input_file, 'r', encoding='utf-8') as f:
         lines = f.readlines()

    # cols = ['CSN','CHTNO','type of note', 'txt']
    a_dict = {}
    a_dict['CSN'] = lines[0].replace('\n','')
    a_dict['CHTNO'] = lines[1].replace('\n','')
    a_dict['txt'] = ''.join(lines[2:])  
  
    out_file = lines[0].replace('\n','')+'outputaiinf'
    
    logging.info(out_file)
    print('輸出檔案:'+out_file)
    
    note_text0 = pd.DataFrame.from_dict([a_dict])
        
    # load key
    key_file = os.path.join(dirname,'cc_keys.csv')
    assert os.path.exists(key_file), 'error!, cc keys must exist'
    allkeys = pd.read_csv(key_file, on_bad_lines='skip')           
    series_keys = allkeys[allkeys.columns[0]] # 轉成series格式
    
    # 比對
    cc_ans, matched_keys_per_txt = cc_detect(note_text0,series_keys)
        
    file_cont = ''
    if cc_ans == 'yes': 
            
       file_cont += allkeys.keys()[0].strip()
       file_cont += '\n'
       
       print(matched_keys_per_txt)
       
       matched_keywords = ''
       for i in matched_keys_per_txt[0]:
           
           matched_keywords += i+','
       
       matched_keywords = matched_keywords[0:len(matched_keywords)-1] 
       
       file_cont += matched_keywords
               
    else:
       
       file_cont += 'N'
        
    logging.info(file_cont)           
    print(cc_ans)
    
    # save the results
    # print('hello')
    with open(os.path.join(dirname_file,out_file+'.txt'),'w') as f:
         f.write(file_cont)
            
            

if __name__ == '__main__':
   
   # verify the input 
   print('輸入檔案:{}'.format(sys.argv[1]))
   print('執行檔所在位置:{}'.format(sys.argv[0]))
   
   main(sys.argv[0], sys.argv[1])
   
   
   
   # # 執行的方式 input哪來, output那？資料清理
   #  if not os.listdir(cc_config.input_folder):  
   #     print('輸入資料夾是空的')       
   #     logging.warning('輸入資料夾是空的')
   #  else:  
       
   #     meta_data = ['12340593',33456987,'手術紀錄']
        
       # main(cc_config, meta_data)
       
     


 