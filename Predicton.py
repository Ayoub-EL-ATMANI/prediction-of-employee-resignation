import pandas as pd
import numpy as np

class model:
    def __init__(self,path):
        self.path = path
        da = pd.read_csv(path)
        self.data=da.head(4200)
        self.dt=pd.DataFrame(self.data)
        self.prediction = None

    def zeror(self):
        count_left_dem = list(self.data.left).count("yes")
        count_left_nn = list(self.data.left).count("no")
        co_M = np.array([[count_left_dem,count_left_nn],[0,0]])
        if max([count_left_dem,count_left_nn]) == count_left_dem:
            accury = count_left_dem / (count_left_dem + count_left_nn)
            print(" accury (demissioner) : ",accury)
        else:
            accury = count_left_nn / (count_left_dem + count_left_nn)
            print(" accury (demissioner) : ",accury)

        dec = {'matrix' : co_M,'accury': accury}
        print(dec)
        return dec
 	#frequency table for 2 target (left = yes , left =no)
    def frequencyTable(self,pred):
        freq = pd.crosstab(index=self.data[pred],columns=self.data["left"])
        freq_tab = pd.DataFrame(freq) 
        majority = list()
        errors = list()
        error_class = list()
        for key in freq_tab.index:
            if freq_tab.loc[str(key)][0] < freq_tab.loc[str(key)][1]:
                small = freq_tab.loc[str(key)][0] 
                error_class.append({'no' :  freq_tab.loc[str(key)][0]})
                majority.append({'yes' : freq_tab.loc[str(key)][1] })
                errors.append(small)   
            else:
                small = freq_tab.loc[str(key)][1] 
                error_class.append({'yes':  freq_tab.loc[str(key)][1]}) 
                majority.append({'no' : freq_tab.loc[str(key)][0] })
                errors.append(small) 
        freq_tab['error_class'] = error_class 
        freq_tab['majority_class']=majority
        #freq_tab['errors'] = errors
        #print(freq_tab)
        return freq_tab
   	# calculate error frequency table on algo oneR    
    def calculerrorFre(self,pred):
        freq_tab = model.frequencyTable(self,pred)
        error_freq_tab = 0.0
        minn = list()
        total = 0.0
        for key in freq_tab.index:
            minn.append(min(freq_tab.loc[str(key)][0],freq_tab.loc[str(key)][1]))
            total = total + freq_tab.loc[str(key)][0]+freq_tab.loc[str(key)][1]
        error_freq_tab = sum(minn) / total 
        print(error_freq_tab)     
        return error_freq_tab

    def calculerrorTotale(self,*pred):
        fr = { key : self.calculerrorFre(key) for key in pred}
        print('min : ',min(fr),min(fr.values()))
        return fr

    def minerrortotale(self,*pred):
        fr = { key : self.calculerrorFre(key) for key in pred}
        #print('min : ',min(fr),min(fr.values()))
        dic_min_error = {min(fr) : min(fr.values())}
        #print("hello ",dic_min_error.keys())
        self.prediction = list(dic_min_error)[0]
        
        return dic_min_error

    def freqTableMin(self):
        #min = model.minerrortotale(self,pred)
        assert(self.prediction != None) , "most call minerrortotale function before !!"
        return self.frequencyTable(self.prediction)

    def confusionMatrix(self):
        freqtable = self.freqTableMin()      
        m_yes=0
        m_no =0
        e_yes=0
        e_no=0
        
        for index in freqtable.index:

            if freqtable.loc[index]['no'] > freqtable.loc[index]['yes']:
                m_no+=freqtable.loc[index]['no'] 
                e_yes+=freqtable.loc[index]['yes']
            else:
                m_yes+=freqtable.loc[index]['yes']
                e_no+=freqtable.loc[index]['no']

        return pd.DataFrame([[m_no , e_yes],[e_no , m_yes]] , index=['no' , 'yes'] , columns=['no' , 'yes'])

    def Accuracyonr(self):
        dt = self.confusionMatrix() 
        accuracy = (max(dt['no']['no'],dt['yes']['no']) + max(dt['yes']['yes'],dt['no']['yes'])) / (dt['no']['no'] + dt['yes']['no'] + dt['yes']['yes'] + dt['no']['yes'])
        return accuracy 

    def posvaluepred(self):
        dt = self.confusionMatrix()
        if dt['no']['no'] !=0 or dt['yes']['no'] !=0 :
            pospredval = max(dt['no']['no'],dt['yes']['no']) / (dt['no']['no']+dt['yes']['no'])
        else :
            pospredval = 0        
        return pospredval

    def negvalupred(self):
        dt = self.confusionMatrix()
        if dt['no']['yes'] !=0 or dt['yes']['yes'] !=0 : 
            negpredval = max(dt['no']['yes'],dt['yes']['yes']) / (dt['no']['yes'] + dt['yes']['yes'])
        else:
            negpredval = 0
        return negpredval

    #Accuracy = (a+d)/(a+b+c+d),  Positive Predictive Value  a/(a+b), Negative Predictive Value d/(c+d)
    #Sensitivity a/(a+c) , Specificity d/(b+d)

    def prob_yes(self,pred):
        freq_table=self.frequencyTable(pred)
        probyes = 0.0
        totale  = self.data.left.count()
        probfreq = 0.0
        for key in freq_table.index:
            probfreq = probfreq + freq_table.loc[key][1] 
        probyes = probfreq / totale
        return probyes 
        
    def prob_no(self,pred):
        freq_table=self.frequencyTable(pred)
        probno = 0
        totale  = self.data.left.count()
        probfreq = 0
        for key in freq_table.index:
            probfreq = probfreq + freq_table.loc[key][0] 
        #print("freq",probfreq)
        #print("totale",totale)
        probno = probfreq / totale
        return probno

    def prob_x(self,pred,key):
        freq_table=self.frequencyTable(pred)
        probx = freq_table.loc[key][0] + freq_table.loc[key][1]
        return probx / self.data.left.count()
    #likelihood of predector (factor)      
    def likelihood(self,pred):
        like_lih = pd.crosstab(index=self.data[pred],columns=self.data['left'])
        like_lihood = pd.DataFrame(like_lih).astype(float)
        prob_sum_no = 0.0
        prob_sum_yes  = 0.0
        for key in like_lihood.index:
            prob_sum_no = prob_sum_no + like_lihood.loc[key][0]
            prob_sum_yes = prob_sum_yes + like_lihood.loc[key][1]
        for key in like_lihood.index:
            like_lihood.loc[key][0] = like_lihood.loc[key][0] / prob_sum_no 
            like_lihood.loc[key][1] = like_lihood.loc[key][1] / prob_sum_yes
        #print("somm no , somme yes ",prob_sum_no,prob_sum_yes)
        #print("test test : ",like_lihood.loc['passable'][1])
        return like_lihood
   	# probability x of left equals no 
    def prob_class_x_no(self,pred,key):
        likelihod = self.likelihood(pred)
        pred = likelihod.loc[key][0] * self.prob_no(pred) / self.prob_x(pred,key)
        return pred
    # probability x of left equals yes
    def prob_class_x_yes(self,pred,key):
        likelihod = self.likelihood(pred)
        pred = likelihod.loc[key][1] * self.prob_yes(pred) / self.prob_x(pred,key)
        return pred
    #p(c|x) = p(x1|c) * p(x2|c) *p(x3|c) ... p(xn|c) * p(c)
    # probability of value predector give left = no (employee is continue job in company )  (for example factor salary with value low ) 
    def predf_n(self,pred,key):
        likelihod = self.likelihood(pred)
        if likelihod.loc[key][0] == 0:
            likelihod.loc[key][0] = 1
        pred_x_c = likelihod.loc[key][0]
        return pred_x_c
    # probability of value predector give left = yes (employee is resign)  (for example factor salary with value low ) 	
    def predf_ye(self,pred,key):
        likelihod = self.likelihood(pred)
        if likelihod.loc[key][1] == 0:
            likelihod.loc[key][1] = 1
        pred_x_c = likelihod.loc[key][1]
        return pred_x_c
   	# cette méthode donne est accepter les parametres (facteur et leur valeur) d'une maniere dynamique, mais si en donne plusieurs paramétre la méthode donne une résultas plus correct. (method for probability of employee continue your job (left == no)) 
    def pred_generale_no(self,**pred):
        result = list()
        produit = 1
        for key,value in pred.items():
            result.append(self.predf_n(key,value))
        for i in result:
            produit = produit * i
        no = self.dt.loc[self.dt.left == 'no']    
        pred_n_gn = produit * (len(no) / self.data.left.count())
        return pred_n_gn
  	# cette méthode donne est accepter les parametres (facteur et leur valeur) d'une maniere dynamique, mais si en donne plusieurs paramétre la méthode donne une résultas plus correct. (method for probability of employee resign  (left == yes)) 
    def pred_generale_yes(self,**pred):
        result = list()
        produit = 1
        for key,value in pred.items():
            result.append(self.predf_ye(key,value))
        for i in result:
            produit = produit * i
        yes = self.dt.loc[self.dt.left == 'yes']    
        pred_yes_gn = produit * (len(yes) / self.data.left.count())
        return pred_yes_gn
    


	    


mo = model('data.csv')
#print(mo.minerrortotale('satisfaction_level','last_evaluation','number_project','Work_accident','promotion_last_5years','department','salary'))

#print(mo.freqTableMin())
print("*********************  frequency table ******************")
print(mo.frequencyTable('Work_accident'))
print("********************* likelihood table ******************")
print(mo.likelihood('Work_accident'))
print("********************* probability ******************")

print('P(yes) = ',mo.prob_yes('Work_accident'))
print('P(no) = ',mo.prob_no('Work_accident'))
print('P(x) = ',mo.prob_x('satisfaction_level','tres bien'))
print('P(no|x) = ',mo.prob_class_x_no('number_project','medium'))
print('P(yes|x) = ',mo.prob_class_x_yes('number_project','medium'))
print('p(x|no) = ',mo.predf_n('Work_accident','yes'))
print('p(x|yes) = ',mo.predf_ye('Work_accident','yes'))
#assez bien,bien,low,no,no,support,low
no = mo.pred_generale_no(satisfaction_level='assez bien',last_evaluation='bien',number_project='low',Work_accident='no',promotion_last_5years='no',department='support',salary='low')
yes = mo.pred_generale_yes(satisfaction_level='assez bien',last_evaluation='bien',number_project='low',Work_accident='no',promotion_last_5years='no',department='support',salary='low')
continu = no / (no+yes)
demissioner = yes /(yes+no)
print('Probability of continue  = ',continu)
print('Probability of resign = ',demissioner)

if continu > demissioner : 
    print()
    print(' employee can continue ')
elif continu < demissioner : 
    print()
    print(' the employee can resign')    
else : 
    print()
    print(' the prediction of resignation and continuity of employee are equal ')    
 
#satisfaction_level,last_evaluation,number_project,Work_accident,promotion_last_5years,department,salary

"""
print(mo.confusionMatrix())

print("*********************")

print("accuracy oneR : ",mo.Accuracyonr())
print(" pos value : ",mo.posvaluepred())
print(" neg value : ",mo.negvalupred())
"""     
        

