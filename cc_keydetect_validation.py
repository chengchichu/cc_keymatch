#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:41:21 2022

@author: anpo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:04:13 2022

@author: anpo
"""

import numpy as np
import pandas as pd
import re 
from fuzzysearch import find_near_matches
import matplotlib.pyplot as plt
import math
import itertools
import os

# functions

# def extract_cc_info(gb_data, division_code):
# # 某科
#     gb_data['出院科別'] = gb_data['出院科別'].astype(str)
    
#     div_case = gb_data['出院科別'].isin(division_code)
  
#     note_text = gb_data.loc[div_case,['CSN','CHTNO','ABSTO', 'ABSTO1', 'ABSTO2', 'ABSTA', 'ABSTA1',
#        'ABSTA2', 'ABSTP', 'ABSTP1', 'ABSTP2','DSTEXC','DSTOP', 'DSTTXPR','ADMPRET', 'ADMPRET1','ADMPASS1', 'ADMPASS2', 'ADMIMPR', 'ADMPLAN']]
    
#     # CSN住院號不重複
#     try:
#        div_gt = gb_data.loc[div_case,'RA次分類(CC項次分類)']
#     except:
#        # div_gt = gb_data.loc[div_case,'RA次分類']
#        div_gt = gb_data.loc[div_case,'"RA分類"'] 
#     loc_gt = gb_data.loc[div_case,'院區']
#     idt = gb_data.loc[div_case,['CSN', 'CHTNO']].reset_index()
#     return div_gt, loc_gt, note_text,idt
 
def extract_cc_info2(gb_data):
# 某科

    note_text = gb_data.loc[:,['CSN','CHTNO','ABSTO', 'ABSTO1', 'ABSTO2', 'ABSTA', 'ABSTA1',
       'ABSTA2', 'ABSTP', 'ABSTP1', 'ABSTP2','DSTEXC','DSTOP', 'DSTTXPR','ADMPRET', 'ADMPRET1','ADMPASS1', 'ADMPASS2', 'ADMIMPR', 'ADMPLAN']]
    
    # CSN住院號不重複
    try:
       div_gt = gb_data.loc[:,'RA次分類(CC項次分類)']
    except:
       # div_gt = gb_data.loc[:,'RA次分類']
       div_gt = gb_data.loc[:,'"RA分類"'] 
    loc_gt = gb_data.loc[:,'院區']
    idt = gb_data.loc[:,['CSN', 'CHTNO']].reset_index()
    return div_gt, loc_gt, note_text,idt    

def keyword_match2(note_text, allkeys, use_fuzz=False):

    id_part = note_text[['CSN', 'CHTNO']]   
    txt_part = note_text.drop(columns = ['CSN', 'CHTNO'])
    # 清理text
    note_text_dropna = []
    # assert(len(txt_part)==len(div_gt)) 
    note_text_merge = []
    for i in range(len(txt_part)):
        vv = ''
        v = txt_part.iloc[i,:].dropna()
        note_text_dropna.append(v)
        for i in range(len(v)):
            vv = vv+v[i]
        
        vv = vv.replace('.','')
        vv = vv.replace(',','')
        # 
        vv = vv.replace('\r','')
        vv = vv.replace('\n','')
        vv = re.sub(' +','',vv)  
        note_text_merge.append(vv)
    
    id_part['txt_flat'] = note_text_merge
    # txt_flat = pd.Series(note_text_merge)
    # id_part = id_part.append(txt_flat, ignore_index = True)
    id_part = id_part.reset_index()
         
    # tn_idc = []
    # matched_keys_per_txt = []
    # whether_get_keyword = np.zeros((1,len(note_text_merge))).astype(bool)[0]
    Nans = []
    yes_id = []
    for idc, rows in id_part.iterrows():
        per_txt = rows['txt_flat']
        matched_keys = []
        cnt2 = 0
        # 清理key
        for index, value in allkeys.items():
            # value = value.replace('.','')
            # value = value.replace(',','')
            # remove extra space 
            value = re.sub(' +','',value)
            # re search
            if use_fuzz:
            # fuzzy search 
                b = find_near_matches(value.lower(),per_txt.lower(), max_l_dist=1)               
            # print(b)
                if b!=[]:
                  matched_keys.append(value)
                  cnt2+=1
            else: 
                b = re.search(value.lower(),per_txt.lower())        
                if b!=None: 
                  matched_keys.append(value)
                  cnt2+=1
                  
        if cnt2>0:
           ans = 'yes'
           yes_id.append(idc)
        else:
           ans = 'no'
        Nans.append(ans)
        yes_ids = id_part.loc[yes_id,['CSN', 'CHTNO']]
    return Nans, yes_ids


def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n


def cc_metrics(counts, note_text,div_gt): 
    cnt = 0
    f1s = []
    sens = []
    for k,v in counts.iteritems():
        key1 = pd.Series([k])
        _, _, _, r = keyword_match(note_text, key1, div_gt, use_fuzz=False)    
        try:
          sen = r[0]/(r[0]+r[1])
          acc = (r[0]+r[3])/sum(r)
          prc = r[0]/(r[0]+r[2])      
          f1 = 2*prc*sen/(prc+sen)
          f1 = truncate(f1, 2)
        except:
          f1 = 0
          sen = 0
          prc = 0
          acc = 0
           
        f1s.append(str(f1))
        sen = truncate(sen, 2)
        sens.append(sen)
        cnt+=1
    
    return sens, f1s  


def data_type_convert(df):
    for index, row in df.iterrows():
        try:
           df.loc[index, 'CSN'] = str(int(float(row['CSN'])))
        except:
           df.loc[index, 'CSN'] = str(row['CSN'])    
           
        df['CHTNO'] = df['CHTNO'].astype(int)
    return df   

# main start here +=================


# cc_ground = pd.read_excel('2021年1-9月CC項次類別個案明細(20220531).xlsx')

# 先把重要的parse出來. 關鍵字, cc類別
import ccdata_parse

allkeys = ccdata_parse.allkeys
cc_type = ccdata_parse.cc_type
division_code = ccdata_parse.division_code

cc_ground0 = ccdata_parse.cc_ground0
cc_ground1 = ccdata_parse.cc_ground1
# cc_ground = ccdata_parse.cc_ground
# 去掉gt重複筆數 
# cc_ground_uni = cc_ground.drop_duplicates(subset=['住院號','病歷號','住院日','出院日'])
# 因為要比對, 所以用出院日串註記日 (但兩者應該不同先不要這樣)



## 醫科來的病歷, 串接,  

note_d0 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/2021 - 2022 June/hemo/ricdaitsipd.csv', skiprows = 1)  # absto
note_e0 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/2021 - 2022 June/hemo/ricdaitsipd2.csv', skiprows = 1) # dstop
note_f0 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/2021 - 2022 June/hemo/ricdaitsipd3.csv', skiprows = 1) # ad note

note_d1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd_5.csv', skiprows = 1)  # absto
note_e1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd2_5.csv', skiprows = 1) # dstop
note_f1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd3_5.csv', skiprows = 1) # ad note

note_d0 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/2021 - 2022 June/ricdaitsipd_d.csv', skiprows = 1)  # absto
note_e0 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/2021 - 2022 June/ricdaitsipd2_d.csv', skiprows = 1) # dstop
note_f0 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/2021 - 2022 June/ricdaitsipd3_d.csv', skiprows = 1) # ad note

note_d1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd_8.csv', skiprows = 1)  # absto
note_e1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd2_8.csv', skiprows = 1) # dstop
note_f1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd3_8.csv', skiprows = 1) # ad note

# # 5 直肛 6 骨科 7泌尿 8腦神經外  要調整喔 不然會抓不到 e.g. 骨科 ricdaitsipd_6,  ricdaitsipd_b



# note_d = note_d0
# note_e = note_e0
# note_f = note_f0

def pool_data(cc_ground, note_d, note_e, note_f):

    cc_ground_re = cc_ground.rename(columns={'住院號':'CSN','病歷號':'CHTNO'})    

    note_d_uni = note_d
    note_d_uni_ = note_d_uni[~note_d_uni['CSN'].isna() & ~note_d_uni['CHTNO'].isna()] 
    note_d_uni_ = data_type_convert(note_d_uni_)
    # note_d_uni_.to_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/converted_ipd.csv')
    # note_d_uni_ = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/converted_ipd.csv')
    
    note_e_uni = note_e
    note_e_uni_ = note_e_uni[~note_e_uni['CSN'].isna() & ~note_e_uni['CHTNO'].isna()] 
    note_e_uni_ = data_type_convert(note_e_uni_)
    # note_e_uni_ = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/converted_ipd2.csv')
    
    # note_f_uni = note_f.drop_duplicates(subset=['CSN','CHTNO']) # 
    note_f_uni = note_f
    note_f_uni_ = note_f_uni[~note_f_uni['CSN'].isna() & ~note_f_uni['CHTNO'].isna()] 
    note_f_uni_ = data_type_convert(note_f_uni_)
    # note_f_uni_ = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/converted_ipd3.csv')
    
    cc_ground_re = data_type_convert(cc_ground_re)   
    # cc_ground_re.to_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/cc_ground.csv')
    # cc_ground_re = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/cc_ground.csv')
    
    DATA = []
    DATA.append(cc_ground_re)
    DATA.append(note_d_uni_)
    DATA.append(note_e_uni_)
    DATA.append(note_f_uni_)
    
    # gt上只取第一筆喔
    gb_data = DATA[0].groupby(['CSN','CHTNO'], as_index=False).first()[['CSN','CHTNO']]
    for aa in DATA:
        if 'RDDAT' in aa.keys():
            for key in aa.keys():
                if key not in gb_data.keys():
                    tmp = aa.dropna(subset=[key]).sort_values(by='RDDAT').groupby(['CSN','CHTNO'], as_index=False).last()
                    gb_data = gb_data.merge(tmp[['CSN',key]], on='CSN', how='left')
        else:
            # note上取最後一筆喔 
            tmp = aa.groupby(['CSN','CHTNO'], as_index=False).last()
            gb_data = gb_data.merge(tmp[['CSN']+[ii for ii in tmp.keys() if ii not in gb_data.keys()]], on='CSN', how='left')
    
    return gb_data
     

gb_data0 = pool_data(cc_ground0, note_d0, note_e0, note_f0)
gb_data1 = pool_data(cc_ground1, note_d1, note_e1, note_f1)
# data split for train and test
# gb_data0 = gb_data
# train_data = gb_data.loc[gb_data['年'].isin([2018,2019]),:]
# test_data = gb_data.loc[gb_data['年'].isin([2020]),:]
test_data = pd.concat([gb_data0, gb_data1])



# test_div_gt, _, test_note_text, _ = extract_cc_info2(test_data[test_data['"RA分類"'] != 'Infection'])
# test_div_gt, _, test_note_text, _ = extract_cc_info2(test_data[test_data['"RA分類"'] != 'Disruption'])
test_div_gt, _, test_note_text, _ = extract_cc_info2(test_data)
    
# tn_idc, note_text_dropna, matched_keys_per_txt, _ = keyword_match(note_text, allkeys, div_gt, use_fuzz=False, print_tag = True)
# _, _, _, r = keyword_match(test_note_text, allkeys, test_div_gt, use_fuzz=False, print_tag = True)

def infer(cc_type, note_text, allkeys, div_gt = pd.DataFrame({'A' : []})):

    print('infer: {}'.format(cc_type))
    Nans, yes_ids = keyword_match2(note_text, allkeys, use_fuzz=False)
    
    if not div_gt.empty:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        GY = div_gt == cc_type
        tn_idc = []
        for idc, i in enumerate(Nans):
            if (i == 'yes') & (GY.iloc[idc] == True):
                tp+=1    
            if (i == 'no') & (GY.iloc[idc] == True):
                fn+=1 
                tn_idc.append(idc)
            if (i == 'yes') & (GY.iloc[idc] == False):
                fp+=1    
            if (i == 'no') & (GY.iloc[idc] == False):
                tn+=1   
               
        sen = tp/(tp+fn)
        prc = tp/(tp+fp)
        print(div_gt.value_counts())
        print('tp:{}'.format(tp))
        print('tn:{}'.format(tn))
        print('fp:{}'.format(fp))
        print('fn:{}'.format(fn))
        print('sen:{}'.format(sen))
        print('prc:{}'.format(prc))        

    
    # 設定位置把歷史檔吐在那裡
    if os.listdir('/home/anpo/Desktop/cc_nlp/history'):
       hist = pd.read_csv('/home/anpo/Desktop/cc_nlp/history/cc_inference_record.csv')
       hist['CHTNO']
       Hans = note_text['CHTNO'].isin(hist['CHTNO'].values)
              # 比對 yes_id跟歷史有
       current = note_text['CHTNO']
              
    else: 
       yes_ids.to_csv('/home/anpo/Desktop/cc_nlp/history/cc_inference_record.csv')    

    return Nans, Hans, sen

#關鍵字偵測, Nans, 是否抓到, Hans, 上次是否抓到 不用顯示類別, 只用在單一科別

# infer(cc_type, note_text, allkeys, div_gt)

# 來測試吧
allkeys = allkeys.reset_index(drop=True)

allkeys0 = allkeys.iloc[[1,2,4,10,11,12]]  # 直肛
allkeys0 = allkeys.iloc[[26,27,28,29,30,6,7,8,9,10,11,12,13,15,4,5,0,1,18,21,31,32]] # disruption
allkeys0 = allkeys
allkeys0 = allkeys.iloc[[4,5,18,13,15,12,11,7,14,19]]  # transplant
allkeys0 = allkeys.iloc[[62,29,60,55,63,41,2,3,4,24,31,39,64,25,20,34,42,16,30,9,47,21,26,6,8,11]]  # mechanism
Nans, Hans = infer(cc_type, test_note_text, allkeys0, test_div_gt)


#
sens = []
LOCS = ['LNK', 'SHK','KEL','JIA'] 
for ic in LOCS:
    aa = test_note_text[test_data['LOC'] == ic]
    bb = test_div_gt[test_data['LOC'] == ic]
    
    Nans, Hans,sen = infer(cc_type, aa, allkeys0, bb)
    sens.append(sen)


a = test_data['LOC'].value_counts()

plt.imshow(np.array([[114,31],[1,55]]))
plt.colorbar()
# test data preprocess
# cc_ground_test = pd.read_excel('/home/anpo/Desktop/cc_nlp/2021年1-9月CC項次類別個案明細(20220531).xlsx', sheet_name = '本次1-9月個案明細(偵測用)')
# cc_ground_test_re = cc_ground_test.rename(columns={'住院號':'CSN','病歷號':'CHTNO'})
# note_d1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd.csv', skiprows = 1)  # absto
# note_e1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd2.csv', skiprows = 1) # dstop
# note_f1 = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitsipd3.csv', skiprows = 1) # ad note

# note_d1_ = note_d1[~note_d1['CSN'].isna()]
# note_e1_ = note_e1[~note_e1['CSN'].isna()]
# note_f1_ = note_f1[~note_f1['CSN'].isna()]

# cc_ground_test_re = data_type_convert(cc_ground_test_re)  
# note_d1 = data_type_convert(note_d1_)
# note_e1 = data_type_convert(note_e1_)
# note_f1 = data_type_convert(note_f1_)
# Test_DATA = []
# Test_DATA.append(cc_ground_test_re)
# Test_DATA.append(note_d1)
# Test_DATA.append(note_e1)
# Test_DATA.append(note_f1)
# # gt上只取第一筆喔
# test_gb_data = Test_DATA[0].groupby(['CSN','CHTNO'], as_index=False).first()[['CSN','CHTNO']]
# for aa in Test_DATA:
#     if 'RDDAT' in aa.keys():
#         for key in aa.keys():
#             if key not in test_gb_data.keys():
#                 tmp = aa.dropna(subset=[key]).sort_values(by='RDDAT').groupby(['CSN','CHTNO'], as_index=False).last()
#                 test_gb_data = test_gb_data.merge(tmp[['CSN',key]], on='CSN', how='left')
#     else:
#         # note上取最後一筆喔 
#         tmp = aa.groupby(['CSN','CHTNO'], as_index=False).last()
#         test_gb_data = test_gb_data.merge(tmp[['CSN']+[ii for ii in tmp.keys() if ii not in test_gb_data.keys()]], on='CSN', how='left')

# test_div_gt, _, test_note_text, _  = extract_cc_info(test_gb_data, division_code)


# Nans, Hans = infer(cc_type, test_note_text, allkeys, test_div_gt)




# 其他分析


# _, _ , test_matched_keys_per_txt ,_ =  keyword_match(test_note_text, allkeys, test_div_gt, use_fuzz=False, print_tag = True)
# 院區分開分析

for loci in loc_gt.unique():
    note_loc = note_text[loc_gt == loci]
    gt_loc = div_gt[loc_gt == loci]
    _, _, _, r = keyword_match(note_loc, allkeys, gt_loc, use_fuzz=False)
    sen = r[0]/(r[0]+r[1])
    print('院區:{}'.format(loci) )
    print('sen:{}'.format(sen))
  
   
# idv key evaluation    
# 抓出關鍵字
extract_keys = pd.DataFrame(matched_keys_per_txt)
k = div_gt == cc_type

keys = extract_keys[k.values]

for i in range(keys.shape[1]):
    b = keys.iloc[:,i]
    if i == 0:
       base = b
    else:
       base = pd.concat([base,b])

counts = base.value_counts() # 關鍵字table


# extract_keys = pd.DataFrame(test_matched_keys_per_txt)
# k = test_div_gt == cc_type






# prcent = counts.values/sum(counts.values)
# 個別關鍵字權重



sens, f1s = cc_metrics(counts, note_text,div_gt) 

# from sklearn.metrics import fbeta_score
def selection_metrics_based(test_note_text, test_div_gt, metric, counts):        
    table = pd.DataFrame({'key':counts.index,'metric':metric})
    table_sorted = table.sort_values(by=['metric'],ascending=False).reset_index()
    print(table_sorted)  
 
    test_f1s = []
    test_sen = []
    test_prc = []
    rs = []
    for i in range(len(table_sorted)):
        if i != 0:
           rng = table_sorted.loc[range(0,i),'key']
        else:
           rng = table_sorted.loc[0,'key'] 
        selected_keys = pd.Series(rng)
        print(selected_keys)
        _,_,_,r = keyword_match(test_note_text, selected_keys, test_div_gt, use_fuzz=False, cnt_threshold=1)
        try:
            sen = r[0]/(r[0]+r[1])
            sep = r[3]/(r[3]+r[2])
            prc = r[0]/(r[0]+r[2])
            # try:        
            # f1 = 2*prc*sen/(prc+sen)
            fbeta = r[0]/(r[0]+0.1*r[2]+0.9*r[1])
        except:
            sen = 0
            sep = 0
            fbeta = 0
            prc = 0
            
        test_f1s.append(fbeta)
        test_sen.append(sen)
        test_prc.append(prc)
        rs.append(r)

    # mv = max(test_f1s) # appear to be not optimal when prc is extremely low
    # print(test_f1s)
    # print(test_sen)
    # print(test_sep)
    # idc = test_f1s.index(mv)
    # current_sen = test_sen[idc]
    # current_sep = test_sep[idc]
    
    # rng = table_sorted.loc[range(0,idc),'key']
    # selected_keys = pd.Series(rng)
    
    return table_sorted, test_sen, test_prc, rs


table_sorted, test_sen, test_prc, rs = selection_metrics_based(test_note_text, test_div_gt, sens, counts)

fig, ax = plt.subplots()
fig = plt.gcf()
fig.set_size_inches(16.5, 8.5)
plt.plot(test_sen)    
plt.plot(test_prc)    

ax.set_xticks(range(len(table_sorted)))
ax.set_xticklabels(table_sorted['key'])
plt.xticks(rotation=90)
ax.legend(['sensitivity/recall','precision'])
# selected_keys,current_sen, current_sep ,max_f1 = selection_metrics_based(test_note_text, test_div_gt, f1s, counts)


# 排序畫圖
fig, ax = plt.subplots()
fig = plt.gcf()
fig.set_size_inches(16.5, 8.5)
    
# xh = len(counts.index)
bar1 = ax.barh(range(len(counts.index)), counts.values)

cnt = 0
for rect in bar1:
    # height = rect.get_height()
    
    # ax.text(counts.values[cnt], rect.get_y(), f1s[0])
    # print(xh-cnt)
    ax.text(counts.values[cnt], cnt, sens[cnt])
    cnt+=1      

ax.set_yticks(range(len(counts.index)))
ax.set_yticklabels(counts.index)




#找出沒抓到的
# full list identifier
#idt
pdi = idt.loc[2, ['CSN','CHTNO']].values
write_txt(pdi, note_text_dropna[2])

# write function
def write_txt(pdi, main_txt):
    with open('/home/anpo/Desktop/cc_nlp/關鍵字沒抓到的cc/'+'CNS_'+str(pdi[0])+'_CHTNO_'+str(pdi[1])+'.txt', 'w') as f:
         for idc, j in main_txt.items():
             f.write('==============================')
             f.write(idc)
             f.write('\n')
             f.write(j)
             f.write('\n')

# 只看沒抓到的

txt = [note_text_dropna[i] for i in tn_idc] 

identifier = idt.loc[tn_idc,['CSN','CHTNO']].values

for idc, i in enumerate(txt):
    pdi = identifier[idc]
    write_txt(pdi, i)
    

ids = pd.DataFrame(identifier,columns = {'CSN','CHTNO'})
ids.to_excel('true_negative_IDs.xlsx')

# 

# def keyword_match(note_text, allkeys, div_gt, use_fuzz=False, print_tag=False, cnt_threshold=0):

#     # 清理text
#     note_text_dropna = []
#     assert(len(note_text)==len(div_gt)) 
#     note_text_merge = []
#     for i in range(len(note_text)):
#         vv = ''
#         v = note_text.iloc[i,:].dropna()
#         note_text_dropna.append(v)
#         for i in range(len(v)):
#             vv = vv+v[i]
        
#         vv = vv.replace('.','')
#         vv = vv.replace(',','')
#         # 
#         vv = vv.replace('\r','')
#         vv = vv.replace('\n','')
#         vv = re.sub(' +','',vv)  
#         note_text_merge.append(vv)
    
#     tp = 0
#     tn = 0
#     fp = 0
#     fn = 0
    
#     tn_idc = []
#     matched_keys_per_txt = []
#     whether_get_keyword = np.zeros((1,len(note_text_merge))).astype(bool)[0]
#     for idc, per_txt in enumerate(note_text_merge):
#         matched_keys = []
#         cnt2 = 0
#         # 清理key
#         for index, value in allkeys.items():
#             value = value.replace('.','')
#             value = value.replace(',','')
#             # remove extra space 
#             value = re.sub(' +','',value)
#             # re search
#             if use_fuzz:
#             # fuzzy search 
#                 b = find_near_matches(value.lower(),per_txt.lower(), max_l_dist=1)               
#             # print(b)
#                 if b!=[]:
#                   matched_keys.append(value)
#                   cnt2+=1
#             else: 
#                 b = re.search(value.lower(),per_txt.lower())        
#                 if b!=None: 
#                   matched_keys.append(value)
#                   cnt2+=1
         
#          # 假如是infection, 一定要搭配其他的字吧,不跳提醒cnt = 0, 
#         if (cnt2 == 1): 
#             if (matched_keys[0] == 'infection'): 
#                 cnt2 = 0

            
#                 # find   
#         if cnt2>0:
#            whether_get_keyword[idc] = True    
        
#         # 比對原則 
        
        
#         # classical, 是否有抓到key, 是否為某個特定的cc type
#         if (cnt2>cnt_threshold) & (div_gt.values[idc] == cc_type):
#             tp+=1    
#         if (cnt2<=cnt_threshold) & (div_gt.values[idc] == cc_type):
#             tn+=1 
#             tn_idc.append(idc)
#         if (cnt2>cnt_threshold) & (div_gt.values[idc] != cc_type):
#             fp+=1    
#         if (cnt2<=cnt_threshold) & (div_gt.values[idc] != cc_type):
#             fn+=1
           

#         matched_keys_per_txt.append(matched_keys)
    
#     sen = tp/(tp+tn)
#     prc = tp/(tp+fp)
#     if print_tag == True:
#        print(div_gt.value_counts())
#        print('tp:{}'.format(tp))
#        print('tn:{}'.format(tn))
#        print('fp:{}'.format(fp))
#        print('fn:{}'.format(fn))
#        print('sen:{}'.format(sen))
#        print('prc:{}'.format(prc))
#     results = [tp,tn,fp,fn]
 
#     return tn_idc, note_text_dropna, matched_keys_per_txt, results