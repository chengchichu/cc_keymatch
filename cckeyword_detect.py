#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:04:13 2022

@author: anpo
"""

import numpy as np
import pandas as pd
import re 



## cc 基隆給的ground truth
cc_ground = pd.read_excel('/home/anpo/Desktop/cc_nlp/2018-2020年RA次分類科別筆數分析(2022.03.03-含科代號).xlsx')
cc_division = pd.read_excel('/home/anpo/Desktop/cc_nlp/2018-2020年RA次分類科別筆數分析(2022.03.03-含科代號).xlsx', sheet_name = '科代號對照')
cc_division['科別'].value_counts()

div_ = '骨科'
div_sheet_name = '骨科CC調查-Infection基隆'
col_name = '基隆骨科初版/調查其他院區該項衍生書寫內容'
cc_type = 'Infection'

div_ = '骨科'
div_sheet_name = '骨科CC調查-Disruption(土城)'
col_name = '基隆骨科初版/調查其他院區該項衍生書寫內容'
cc_type = 'Disruption'


# div_ = '腦神經外科' #12.8%
# div_sheet_name = '神外ＣＣ調查-Ｍechanism(嘉義)'
# col_name = '嘉義神經外科初版/調查其他院區該項衍生書寫內容'
# cc_type = 'Mechanism'



# div_ = '直腸肛門科' # 83.5%  243/291
# div_sheet_name = '直腸肛門外科CC調查-Hemorrage(高雄)'
# col_name = '高雄院區直腸肛門外科初板/調查其他院區該項衍生書寫內容'
# cc_type = 'Hemorrage'



# div_ = '泌尿科'  # 88.8%  199/224
# div_sheet_name = '泌尿科CC調查-Transplant(林口)'
# col_name = '林口泌尿科初版/其他院區該項衍生書寫內容'
# cc_type = 'Transplant'

division_code = cc_division['出院科別代號'][cc_division['科別'] == div_].values
# n = len(cc_ground['RA次分類(CC項次分類)'])

# 去掉gt重複筆數 
# cc_ground_uni = cc_ground.drop_duplicates(subset=['住院號','病歷號','住院日','出院日'])
# 因為要比對, 所以用出院日串註記日 (但兩者應該不同先不要這樣)
cc_ground_re = cc_ground.rename(columns={'住院號':'CSN','病歷號':'CHTNO'})


## 醫科來的病歷 
# note_d = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaiipd.csv',skiprows = 1)
# note_d = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaiipd.csv',on_bad_lines='skip')
# note_d = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitnipd1.csv', sep='delimiter', header=None, skiprows = 2)
# note_d = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitnipd1.csv', sep='delimiter')
# note_d = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitnipd1.csv', sep='\t')
note_d = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitnipd.csv', skiprows = 1)  # absto
note_e = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitnipd2.csv', skiprows = 1) # dstop
note_f = pd.read_csv('/home/anpo/Desktop/cc_nlp/cc_from_sql_0308/ricdaitnipd3.csv', skiprows = 1) # ad note


def data_type_convert(df):
    for index, row in df.iterrows():
        try:
           df.loc[index, 'CSN'] = str(int(float(row['CSN'])))
        except:
           df.loc[index, 'CSN'] = str(row['CSN'])    
           
        df['CHTNO'] = df['CHTNO'].astype(int)
    return df    

# note_d_uni = note_d.drop_duplicates(subset=['CSN','CHTNO']) # 
note_d_uni = note_d
note_d_uni_ = note_d_uni[~note_d_uni['CSN'].isna() & ~note_d_uni['CHTNO'].isna()] 
note_d_uni_ = data_type_convert(note_d_uni_)

# note_e_uni = note_e.drop_duplicates(subset=['CSN','CHTNO']) # 
note_e_uni = note_e
note_e_uni_ = note_e_uni[~note_e_uni['CSN'].isna() & ~note_e_uni['CHTNO'].isna()] 
note_e_uni_ = data_type_convert(note_e_uni_)

# note_f_uni = note_f.drop_duplicates(subset=['CSN','CHTNO']) # 
note_f_uni = note_f
note_f_uni_ = note_f_uni[~note_f_uni['CSN'].isna() & ~note_f_uni['CHTNO'].isna()] 
note_f_uni_ = data_type_convert(note_f_uni_)

cc_ground_re = data_type_convert(cc_ground_re)
   

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


# A = cc_ground_uni.filter(items=['CSN','CHTNO'])
# B = note_d_uni_.filter(items=['CSN','CHTNO'])
# C = note_e_uni_.filter(items=['CSN','CHTNO'])
# D = note_f_uni_.filter(items=['CSN','CHTNO'])

# A = cc_ground_uni.loc[:,['CSN','CHTNO']]
# B = note_d_uni_.loc[:,['CSN','CHTNO']]
# C = note_e_uni_.loc[:,['CSN','CHTNO']]
# D = note_f_uni_.loc[:,['CSN','CHTNO']]
# df = pd.DataFrame({'Animal': ['Falcon', 'Falcon','Parrot', 'Parrot'],'Max Speed': [380., 370., 24., 26.],'driver':['a','b','c','d']})           

# def find_common(A, B):
#     A_part_index = pd.merge(A.reset_index(), B, how='inner').set_index('index')
#     B_part_index = pd.merge(B.reset_index(), A, how='inner').set_index('index')
#     # A_part = A.loc[A_part_index.index,:]
#     # B_part = B.loc[B_part_index.index,:]    
#     # A_part_index = A_part_index[~A_part_index.index.duplicated(keep='first')]
#     # B_part_index = B_part_index[~B_part_index.index.duplicated(keep='first')]   
#     assert(len(A_part_index.index) == len(B_part_index.index))
#     return A_part_index.index, B_part_index.index

# match_b, match_c = find_common(B, C)
# prog_note = note_d_uni_.loc[match_b,['CSN','CHTNO','ABSTO',
#         'ABSTO1', 'ABSTO2', 'ABSTA', 'ABSTA1', 'ABSTA2', 'ABSTP', 'ABSTP1',
#         'ABSTP2']]
# dsop_note = note_e_uni_.loc[match_c,['DSTOP', 'DSTTXPR']]

# # 合併
# note_bc = pd.concat([prog_note.reset_index(), dsop_note.reset_index()], axis=1)
# # 再來比一次, 跟ad note
# tmp = note_bc.loc[:,['CSN','CHTNO']]

# match_bc, match_d = find_common(tmp, D)

# note_bc_sub = note_bc.loc[match_bc,:]
# ad_note = note_f_uni_.loc[match_d,['ADMPRET', 'ADMPRET1','ADMPASS1', 'ADMPASS2', 'ADMIMPR', 'ADMPLAN']]

# note_bc_sub = note_bc_sub.drop(['index'],axis=1)
# note_all = pd.concat([note_bc_sub.reset_index(), ad_note.reset_index()], axis=1)

# N = note_all.loc[:,['CSN','CHTNO']]

# matched_ccgt_index, matched_note_index = find_common(A, N)


# 某科
div_case = gb_data['出院科別'].isin(division_code)
# div_case_note_idx = []
# div_case_gt_idx = []
# for idc, i in enumerate(matched_ccgt_index):
#     if i in div_case.index: # 是某科
#        div_case_note_idx.append(matched_note_index[idc])
#        div_case_gt_idx.append(i)

note_text = gb_data.loc[div_case,['ABSTO', 'ABSTO1', 'ABSTO2', 'ABSTA', 'ABSTA1',
       'ABSTA2', 'ABSTP', 'ABSTP1', 'ABSTP2','DSTOP', 'DSTTXPR','ADMPRET', 'ADMPRET1','ADMPASS1', 'ADMPASS2', 'ADMIMPR', 'ADMPLAN']]
div_gt = gb_data.loc[div_case,'RA次分類(CC項次分類)']

idt = gb_data.loc[div_case,['CSN', 'CHTNO']].reset_index()

# keyword
out = pd.read_excel('/home/anpo/Desktop/cc_nlp/0-合併症書寫內容調查-彙總版20220307.xlsx',sheet_name = div_sheet_name)
allkeys = out[col_name]
allkeys = allkeys.dropna()



def keyword_match(note_text, allkeys, div_gt):

    note_text_dropna = []
    assert(len(note_text)==len(div_gt)) 
    note_text_merge = []
    for i in range(len(note_text)):
        vv = ''
        v = note_text.iloc[i,:].dropna()
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
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    tn_idc = []
    
    whether_get_keyword = np.zeros((1,len(note_text_merge))).astype(bool)[0]
    for idc, per_txt in enumerate(note_text_merge):
    
        cnt2 = 0
        for index, value in allkeys.items():
            value = value.replace('.','')
            value = value.replace(',','')
            # remove extra space 
            value = re.sub(' +','',value)
            # print(value.lower())
            
            b = re.search(value,per_txt.lower())
            if b!=None: 
               cnt2+=1
        
        if cnt2>0:
           whether_get_keyword[idc] = True    
        
        if (cnt2>0) & (div_gt.values[idc] == cc_type):
           tp+=1    
        if (cnt2==0) & (div_gt.values[idc] == cc_type):
           tn+=1 
           tn_idc.append(idc)
        if (cnt2>0) & (div_gt.values[idc] != cc_type):
           fp+=1    
        if (cnt2==0) & (div_gt.values[idc] != cc_type):
           fn+=1 
    
    print(div_gt.value_counts())
    print('tp:{}'.format(tp))
    print('tn:{}'.format(tn))
    print('fp:{}'.format(fp))
    print('fn:{}'.format(fn))
 
    return tn_idc, note_text_dropna
    
tn_idc, note_text_dropna = keyword_match(note_text, allkeys, div_gt)



# 只看沒抓到的

txt = [note_text_dropna[i] for i in tn_idc] 

identifier = idt.loc[tn_idc,['CSN','CHTNO']].values

for idc, i in enumerate(txt):
    pdi = identifier[idc]
    with open('/home/anpo/Desktop/cc_nlp/關鍵字沒抓到的cc/'+'CNS_'+str(pdi[0])+'_CHTNO_'+str(pdi[1])+'.txt', 'w') as f:
         for idc, j in i.items():
             f.write('==============================')
             f.write(idc)
             f.write('\n')
             f.write(j)
             f.write('\n')
             # f.write('')

ids = pd.DataFrame(identifier,columns = {'CSN','CHTNO'})

ids.to_excel('true_negative_IDs.xlsx')
# pd.read_csv('/home/anpo/Desktop/cc_nlp/0-骨科合併症書寫內容調查-20220214彙總版.xlsx',sheet_name = '骨科CC調查-Infectiom(要填)')

# re.match(value,a)


# remove comma and special character

# 把日期改成西元, 為了跟gt比對
# drop_n = 0
# possible_wrong_date = 0
# ind_to_drop = []
# tmp = note_d_uni_['RDDAT'].astype(int).astype(str)
# for index, value in tmp.items():
#     year = value[0:3]
#     if year in ['105','106','107','108','109','110']: 
#        re_date = str(int(year)+1911)+value[3:]  
#        note_d_uni_.loc[index,'RDDAT'] = int(re_date)
#     elif year in ['201', '202']:
#        print(value)
#        note_d_uni_.loc[index,'RDDAT'] = int('2020'+value[2:])
#        possible_wrong_date+=1
#     else:
#        print(year)
#        drop_n+=1
#        ind_to_drop.append(index)
# note_d_uni_.drop([index]) 

# 格式也跟gt一樣 
# note_d_uni_['RDDAT'] = note_d_uni_['RDDAT'].astype(int)


# # 串接其他資料
# note_e_part = note_e.filter(items=['CSN','CHTNO'])
# note_e_part_ = note_e_part.dropna()

# try:
#     note_e_part_['CSN'] = note_e_part_['CSN'].median().round().astype(int).astype(str)
# except:
#     note_e_part_['CSN'] = note_e_part_['CSN'].astype(str)

# note_e_part_['CHTNO'] = note_e_part_['CHTNO'].astype(int)



# x = note_d_uni_.loc[:,['CSN','CHTNO']]
# y = note_e_part_.loc[:,['CSN','CHTNO']]


# arr = y['CSN'].values
# for i in arr:
#     if isinstance(i, str) == False:
#         print(i)

# arr = y['CHTNO'].values
# for i in arr:
#     if isinstance(i, np.int64) == False:
#         print(i)

# a,b,c = np.intersect1d(x['CHTNO'].values,y['CHTNO'].values,return_indices=True)
# z1 = x.iloc[b,:]['CSN'].values
# z2 = y.iloc[c,:]['CSN'].values

# z3 = []
# for i in z1:
#     if i in z2:
#        z3.append(i)

# A, B = find_common_idx(note_d_uni_.loc[:,['CSN','CHTNO']], note_e_part_)





# 弄成一樣的格式
# tmp = note_d_subset['CSN'].astype(int)
# note_d_subset['CSN'] = tmp
# tmp = note_d_subset['CHTNO'].astype(int)
# note_d_subset['CHTNO'] = tmp

# z = note_d_subset.filter(items=['CSN','CHTNO'])
# b = np.unique(z,axis=0)

# for i in range(10):
#     print(i)
#     print(z[(z['CSN'] == b[i,:][0]) & (z['CHTNO'] == b[i,:][1])])

# mapping the two 



# for i in range(len(note_d_uni_)):
    

#     to_match = note_d_uni_.iloc[0,2:5].values 


#     cc_ground_uni.iloc[0,]

# A = cc_ground_uni.filter(items=['CSN','CHTNO','RDDAT'])
# B = note_d_uni_.filter(items=['CSN','CHTNO','RDDAT'])

# 

# matched_note_index = pd.merge(B.reset_index(), A, how='inner').set_index('index')
# matched_ccgt_index = pd.merge(A.reset_index(), B, how='inner').set_index('index')



# gt = cc_ground_uni.loc[matched_ccgt_index,'RA次分類(CC項次分類)']