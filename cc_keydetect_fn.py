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
import cc_config
import logging# functions

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
        if use_fuzz:      # fuzzy search 
            b = find_near_matches(value.lower(),per_txt.lower(), max_l_dist=1)               
            if b!=[]:
              matched_keys.append(value)
              cnt2+=1
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

def main(cc_config, meta_data):
       
    # A = os.path.join(cc_config.input_folder,os.listdir(cc_config.input_folder)[0])
    # note_text = pd.read_csv(A)
    # note_text.iloc[0]['ABSTO']
    
    cols = ['CSN','CHTNO','type of note', 'txt']
    
    files = os.listdir(cc_config.input_folder)
    for ifile in files:
        imedata = meta_data.copy()
        with open(os.path.join(cc_config.input_folder,ifile)) as f:
             txt = f.read()
    
        imedata.append(txt)
        
        a_dict = {}
        assert len(cols) == len(imedata)
        for i, j in zip(cols,imedata):
            a_dict[i] = j 

        outfile = '病例號 {} 住院號 {} 之 {} 開始共病關鍵字偵測'.format(meta_data[0],meta_data[1],meta_data[2])
        logging.info(outfile)
        print(outfile)
    
        note_text0 = pd.DataFrame.from_dict([a_dict])
        
        # print(note_text0)
        
        # 比對
        cc_ans, matched_keys_per_txt = cc_detect(note_text0)
        
        file_cont = ''
        if cc_ans == 'yes': 
            
           file_cont += '書寫出現關鍵診斷:{}'.format(matched_keys_per_txt)
           file_cont += '\n'
           file_cont += '該病人可能有{}等手術合併症, 提醒您事實介入相關處置'.format('infection')
                 
           # 紀錄檔
           # record = []
           record = note_text0[['CSN','CHTNO','type of note']]
           record['cc_type'] = 'Infection'
           
           # print(note_text0)
           # print(record)
           if  os.listdir(cc_config.history_folder):
               
               # 一律把新紀錄加入
               
               hist_files = pd.read_csv(cc_config.history_file)   
               new_history = pd.concat([hist_files,record])
               
               new_history = new_history.drop_duplicates()
               new_history.to_csv(cc_config.history_file, index=False)
            
           else:
               record.to_csv(cc_config.history_file, index=False)    
               print('first time inference, record is saved')   
            
        else:
           file_cont += '未偵測到任何cc相關文字'
        
        logging.info(file_cont)           
        print(file_cont)
        print()
        # save the results
      
        # print('hello')
        with open(os.path.join(cc_config.output_folder,outfile+'.txt'),'w') as f:
             f.write(file_cont)
             
  
             
        
def cc_detect(note_text):

    assert isinstance(note_text, pd.DataFrame) == True
    # assert 'CSN' in note_text
    # assert 'CHTNO' in note_text
    assert len(note_text.columns) > 2  # 除了identifier 還有其他資料
    
    # load keys
    assert os.path.exists(cc_config.key_file), 'error!, cc keys must exist'
    allkeys = pd.read_csv(cc_config.key_file)           
    series_keys = allkeys[allkeys.columns[0]] # 轉成series格式
    
    # print(series_keys)
     # save the keys
     # allkeys = ccdata_parse.allkeys
     # allkeys = allkeys.reset_index(drop=True)
     # allkeys.to_csv('/home/anpo/Desktop/cc_nlp/keys/mechanism_keys.csv', index=False)
    
     # cc key detect...
    # ans, yes_ids, matched_keys_per_txt = keyword_match2(note_text, series_keys, use_fuzz=False)
    ans, matched_keys_per_txt = keyword_match2(note_text, series_keys, use_fuzz=False)
    # print(yes_ids)
    

    return ans, matched_keys_per_txt
    

if __name__ == '__main__':
   
   # initiate log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
   
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(cc_config.log_folder, 'cc_keyfn_log.log'), filemode='w',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)

    logging.info('共病偵測log開始')

   # 執行的方式 input哪來, output那？資料清理
    if not os.listdir(cc_config.input_folder):  
       print('輸入資料夾是空的')       
       logging.warning('輸入資料夾是空的')
    else:  
       
       meta_data = ['12340593',33456987,'手術紀錄']
        
       main(cc_config, meta_data)
       
     


 # 設定位置把歷史檔吐在那裡
    # if os.listdir(cc_config.history_folder):
    #    hist = pd.read_csv(cc_config.history_file)
    #    hist['CT'] = hist['CSN']+hist['CHTNO'].astype(str)     
    #    note_text['CT'] = note_text['CSN']+note_text['CHTNO'].astype(str)
        
    #    wheather_in_history = note_text['CT'].isin(hist['CT'].values)
        
    #     # report cc, if it was in history
    #    if history_reminder:           
    #       for idc, i in enumerate(Nans):
    #           if wheather_in_history.values[idc]:
    #              Nans[idc] = 'yes'
                  
    #     # 是否把新ID增加          
    #    if Add_ID:  
    #       new_history = pd.concat([hist,yes_ids])
    #       new_history.to_csv(cc_config.history_file, index=False)
           
    # else: 
    #    yes_ids.to_csv(cc_config.history_file, index=False)    
    #    print('first time inference, record is saved')